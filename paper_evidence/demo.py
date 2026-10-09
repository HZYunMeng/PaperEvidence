"""Demo requests contain labels only, never expected values or source coordinates."""
PRESETS = {
    'clip': [
        {'label':'CLIP 的 ImageNet 准确率','method':'CLIP','metric':'ImageNet'},
        {'label':'对照方法的 ImageNet 准确率','method':'Visual N-Grams','metric':'ImageNet'},
    ],
    'bert': [
        {'label':'BERT BASE 的 MRPC 分数','method':'BERT BASE','metric':'MRPC'},
        {'label':'BERT LARGE 的 RTE 分数','method':'BERT LARGE','metric':'RTE'},
    ],
    'transformer': [
        {'label':'base 模型的英德 BLEU','method':'Transformer (base model)','metric':'BLEU / EN-DE'},
        {'label':'big 模型的英法 BLEU','method':'Transformer (big)','metric':'BLEU / EN-FR'},
    ],
    'efficientnet': [
        {'label':'EfficientNet-B0 的 Top-1','method':'EfficientNet-B0','metric':'Top-1 Acc.'},
        {'label':'EfficientNet-B7 的参数量','method':'EfficientNet-B7','metric':'#Params'},
    ],
}
