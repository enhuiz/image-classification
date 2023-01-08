mkdir -p figures/$1
python -m src.plot acc.top1 --output figures/$1/acc.top1.png ${@:2}
python -m src.plot model.loss.ce --output figures/$1/loss.png ${@:2}
