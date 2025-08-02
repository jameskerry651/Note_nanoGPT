"""
简易配置器。这可能不是个好主意。示例用法：
$ python train.py config/override_file.py --batch_size=32
这会先运行 config/override_file.py，然后将 batch_size 覆盖为 32

此文件中的代码将从例如 train.py 中按如下方式运行：
>>> exec(open('configurator.py').read())

因此它不是一个 Python 模块，只是把这段代码从 train.py 中分离出来
此脚本中的代码会覆盖全局变量（globals()）

我知道大家可能不喜欢这个方案，但我真的很讨厌配置的复杂性，也不想给每个变量都加上 config. 前缀。如果有人能提出一个更好的简单 Python 解决方案，我洗耳恭听。
"""


"""
@James: 这个文件的主要作用是灵活处理训练参数，达到既可以通过配置文件修改参数，也可以通过命令行参数修改某个具体参数的目的。
"""
import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # 假设这是一个配置文件的名称
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # 假设这是一个 --key=value 参数
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # 尝试对值进行求值（例如，如果是布尔值、数字等）
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # 如果求值失败，直接使用字符串
                attempt = val
            # 确保新值和全局变量中的值类型匹配
            assert type(attempt) == type(globals()[key])
            # 祈祷一切正常
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
