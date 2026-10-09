import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from paper_evidence.answering import number_tokens,validate_answer,answer_question
from paper_evidence.cloud import CompatibleGenerator
from paper_evidence.config import Settings,load_settings
from paper_evidence.retrieval import BM25,Hit
from test_semantic import chunk


class Opener:
    def __init__(self,data,error=None):
        self.data,self.error,self.calls = data,error,[]

    def open(self,request,timeout):
        self.calls.append((request,timeout))
        if self.error:
            raise self.error
        return io.BytesIO(json.dumps(self.data).encode())


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(base_url="https://model.example/v1",model="fixture-model",api_key="test-placeholder")
        self.chunk = chunk("source","The proposed method achieves 91.2% accuracy.")
        self.answer = {"abstain":False,"claims":[{"text":"准确率为91.2%。","evidence":[{
            "chunk_id":"source","quote":self.chunk.text}]}]}
        self.data = {"choices":[{"finish_reason":"stop","message":{"content":json.dumps(self.answer)}}],
                     "usage":{"prompt_tokens":12,"completion_tokens":18,"total_tokens":30}}

    def test_request_uses_only_question_and_retrieved_text(self):
        opener = Opener(self.data)
        generator = CompatibleGenerator(self.settings,opener)
        result = answer_question("accuracy",BM25([self.chunk]),generator)
        self.assertFalse(result["abstain"])
        self.assertEqual(result["mode"],"compatible-api")
        self.assertEqual(result["usage"]["total_tokens"],30)
        request,timeout = opener.calls[0]
        self.assertEqual(request.full_url,"https://model.example/v1/chat/completions")
        payload = json.loads(request.data)
        evidence = json.loads(payload["messages"][1]["content"])["evidence"]
        self.assertEqual(evidence,[{"chunk_id":"source","text":self.chunk.text}])
        self.assertNotIn("test-placeholder",json.dumps(result))
        self.assertEqual(len(opener.calls),1)

    def test_table_generation_reads_canonical_value_from_pdf_cells(self):
        from paper_evidence.parsing import parse_pdf
        from paper_evidence.chunking import chunk_blocks
        chunks=chunk_blocks(parse_pdf(Path(__file__).resolve().parents[1]/"examples/demo-paper.pdf").blocks)
        table=next(c for c in chunks if c.table)
        response={"abstain":False,"claims":[{"table_value":{"chunk_id":table.id,"row":2,"column":1}}]}
        data={**self.data,"choices":[{"finish_reason":"stop","message":{"content":json.dumps(response)}}]}
        opener=Opener(data)
        result=answer_question("accuracy",BM25(chunks),CompatibleGenerator(self.settings,opener))
        self.assertFalse(result["abstain"])
        self.assertEqual(result["claims"][0]["text"],"Proposed · Accuracy：91.2%")
        payload=json.loads(json.loads(opener.calls[0][0].data)["messages"][1]["content"])
        excerpt=next(e for e in payload["evidence"] if e["chunk_id"]==table.id)
        self.assertEqual(excerpt["table_rows"][2][1],"91.2%")
        self.assertNotIn("text",excerpt)

    def test_http_errors_do_not_retry_or_expose_response(self):
        error = urllib.error.HTTPError("https://model.example",401,"secret response body",{},None)
        opener = Opener(None,error)
        generator = CompatibleGenerator(self.settings,opener)
        with self.assertRaisesRegex(RuntimeError,"HTTP 401") as caught:
            generator("question",[Hit(self.chunk,1)])
        self.assertNotIn("secret response body",str(caught.exception))
        self.assertEqual(len(opener.calls),1)

    def test_truncated_generation_is_rejected(self):
        data = {**self.data,"choices":[{"finish_reason":"length","message":{"content":json.dumps(self.answer)}}]}
        generator = CompatibleGenerator(self.settings,Opener(data))
        result = answer_question("accuracy",BM25([self.chunk]),generator)
        self.assertTrue(result["validation_failed"])
        self.assertTrue(result["abstain"])
        self.assertEqual(result["usage"]["total_tokens"],30)

    def test_malformed_message_is_rejected_without_losing_usage(self):
        for choices in ([None],[{"message":None}]):
            generator = CompatibleGenerator(self.settings,Opener({**self.data,"choices":choices}))
            result = answer_question("accuracy",BM25([self.chunk]),generator)
            self.assertTrue(result["validation_failed"])
            self.assertEqual(result["usage"]["total_tokens"],30)

    def test_oversized_evidence_makes_no_request(self):
        from dataclasses import replace
        opener=Opener(self.data)
        generator=CompatibleGenerator(replace(self.settings,max_evidence_chars=100),opener)
        with self.assertRaises(RuntimeError):
            generator("question",[Hit(chunk("big","text "*100),1)])
        self.assertEqual(opener.calls,[])

    def test_chinese_adjacent_fabricated_number_is_caught(self):
        self.assertEqual(number_tokens("准确率99.2%"),{"99.2%"})
        self.answer["claims"][0]["text"]="准确率为99.2%。"
        with self.assertRaises(ValueError):
            validate_answer(self.answer,[Hit(self.chunk,1)])

    def test_settings_reads_key_from_named_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"config.toml"
            path.write_text('[llm]\napi_key_env="MY_KEY"\nmodel="fixture"\n')
            settings=load_settings(path,{"MY_KEY":"test-placeholder"})
            self.assertEqual(settings.api_key,"test-placeholder")
            self.assertNotIn("test-placeholder",repr(settings))

    def test_credentials_in_endpoint_rejected(self):
        from dataclasses import replace
        for url in ("http://model.example/v1","https://model.example/v1?key=secret","https://user:pass@model.example"):
            with self.assertRaises(ValueError):
                CompatibleGenerator(replace(self.settings,base_url=url))
