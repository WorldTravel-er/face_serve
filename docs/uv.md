# 一、uv安装

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 把 ~/.local/bin 加进 PATH（多数 shell 安装脚本已自动处理）
export PATH="$HOME/.local/bin:$PATH"

pip install uv
```

# 二、安装python

更换国内源：创建uv.toml文件，放入项目根目录下

![image-20260809230259040](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260809230259040.png)

```bash
uv python install 3.11
```

不会污染系统 Python，uv 把它放在 `~/.local/share/uv/python/` 下。

```
uv python list # 列举python环境
```

# 二、uv使用

## 1. 初始化：

```bash
uv init xxx(praxis-tui)
```

生成praxias-tui文件夹，并自动生成如下文件。



<img src="C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260805134131918.png" alt="image-20260805134131918"  />

```bash
uv init
uv init -p 3.12  # 指定Python 3.12版本
```

在当前文件夹下生成上述文件

## 2. 添加依赖：

```bash
uv add numpy
uv pip install numpy==xxxx
```

生成虚拟环境，生成或者修改uv.lock文件，修改pyproject.toml。

<img src="C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260805134426946.png" alt="image-20260805134426946"  />

具体来说，在pyproject.toml中的dependence中会加入这个包

<img src="C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260805140706559.png" alt="image-20260805140706559" style="zoom:80%;" />

uv.lock中会记录这个包以及这个包所依赖的其他包的版本以及来源等信息

<img src="C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260805140814688.png" alt="image-20260805140814688" style="zoom:80%;" />

## 3. 已有项目生成uv.lock：

```bash
uv lock
```

如果项目已经有pyproject.toml， uv 会解析并生成uv.lock文件

## 4. 显示包依赖关系：

```bash
uv tree
```

![image-20260805140046773](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260805140046773.png)

## 5. 删除依赖：

```
uv remove numpy
```

删除这个包和它依赖的包

## 6.更新依赖

```bash
uv sync
```

修改了pyproject.toml, uv.lock, .python-version文件后的更新操作，重新解析

## 7. uv工具管理

```bash
uv tool install ruff
uv tool run ruff check
```

检查代码（包import了但未使用， 变量赋值未使用......）

```
uv tool install mypy
uv tool run mypy
```

检查代码（类型错误......）