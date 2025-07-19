仓颉-Python 互操作
为了兼容强大的计算和 AI 生态，仓颉支持与 Python 语言的互操作调用。Python 的互操作通过 std 模块中的 ffi.python 库为用户提供能力。

目前 Python 互操作仅支持在 Linux 平台使用，并且仅支持仓颉编译器的 cjnative 后端。

Python 的全局资源及使用
提供内建函数类以及全局资源
代码原型：

public class PythonBuiltins {
    ...
}
public let Python = PythonBuiltins()
Python 库提供的接口不能保证并发安全，当对 Python 进行异步调用时（系统线程 ID 不一致）会抛出 PythonException 异常。

在 Python 初始化时，GIL 全局解释器锁基于当前所在 OS 线程被锁定，如果执行的代码所在的 Cangjie 线程（包括 main 所在 Cangjie 线程）在 OS 线程上发生调度（OS 线程 ID 发生变化），Python 内部再次尝试检查 GIL 时会对线程状态进行校验，发现 GIL 状态中保存的 OS 线程 ID 与当前执行的 OS 线程 ID 不一致，此时会触发内部错误，导致程序崩溃。

由于 Python 互操作使用到大量 Python 库的 native 代码，这部分代码在仓颉侧无法对其进行相应的栈保护。仓颉栈保护默认大小为 64KB，在对 Python C API 进行调用过程中，容易造成 native 代码超出默认栈大小，发生溢出，会触发不可预期的结果。建议用户在执行 Python 互操作相关代码前，配置仓颉默认栈大小至少为 1MB：export cjStackSize=1MB 。

使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()
    Python.unload()
    return 0
}
提供 Python 库日志类 PythonLogger
代码原型：

public class PythonLogger <: Logger {
    mut prop level: LogLevel {...}
    public func setOutput(output: io.File): Unit {} // do nothing
    public func trace(msg: String): Unit {...}
    public func debug(msg: String): Unit {...}
    public func info(msg: String): Unit {...}
    public func warn(msg: String): Unit {...}
    public func error(msg: String): Unit {...}
    public func log(level: LogLevel, msg: String): Unit {...}
}
public let PYLOG = PythonLogger()
Logger 类的几点声明：

PythonLogger 实现 Logger 接口仅做打印输出以及打印等级控制，不做日志转储到 log 文件；
setOutput 为空实现，不支持 log 转储文件；
info/warn/error 等接口输出打印以对应前缀开头，其他不做区分；
PythonLogger 默认打印等级为 LogLevel.WARN ；
PYLOG.error(msg) 和 log(LogLevel.ERROR, msg) 接口会抛出 PythonException 异常。
使用示例：

import std.ffi.python.*
import std.log.*

main(): Int64 {
    PYLOG.level = LogLevel.WARN // Only logs of the warn level and above are printed.
    PYLOG.info("log info")
    PYLOG.warn("log warn")
    try {
        PYLOG.error("log error")
    } catch(e: PythonException) {}

    PYLOG.log(LogLevel.INFO, "loglevel info")
    PYLOG.log(LogLevel.WARN, "loglevel warn")
    try {
        PYLOG.log(LogLevel.ERROR, "loglevel error")
    } catch(e: PythonException) {}
    return 0
}
执行结果：

WARN: log warn
ERROR: log error
WARN: loglevel warn
ERROR: loglevel error
提供 Python 库异常类 PythonException
代码原型：

public class PythonException <: Exception {
    public init() {...}
    public init(message: String) {...}
}
PythonException 有以下说明：

PythonException 与被继承的 Exception 除了异常前缀存在差异，其他使用无差异；
当 Python 内部出现异常时，外部可以通过 try-catch 进行捕获，如果不进行捕获会打印异常堆栈并退出程序，返回值为 1。
使用示例：

import std.ffi.python.*
import std.log.*

main(): Int64 {
    try {
        Python.load("/usr/lib/", loglevel: LogLevel.INFO)
    } catch(e: PythonException) {
        print("${e}") // PythonException: "/usr/lib/" does not exist or the file path is invalid.
    }
    return 0
}
提供 Python 库的版本信息类 Version
代码原型：

public struct Version <: ToString {
    public init(major: Int64, minor: Int64, micro: Int64)
    public func getMajor(): Int64
    public func getMinor(): Int64
    public func getMicro(): Int64
    public func getVersion(): (Int64, Int64, Int64)
    public func toString(): String
}
关于 Version 类的几点声明：

Version 版本信息包含三个部分：major version，minor version，micro version。
Version 版本仅通过构造函数进行初始化，一旦定义，后续无法修改。
提供 toString 接口，可以直接进行打印。
提供 getVersion 接口，可以获取版本的 tuple 形式。
使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()
    var version = Python.getVersion()
    print("${version}")
    var tuple_version = version.getVersion()
    Python.unload()
    return 0
}
PythonBuiltins 内建函数类
Python 库的导入和加载
代码原型：

public class PythonBuiltins {
    public func load(loglevel!: LogLevel = LogLevel.WARN): Unit
    public func load(path: String, loglevel!: LogLevel = LogLevel.WARN): Unit
    public func isLoad(): Bool
    public func unload(): Unit
}
public let Python = PythonBuiltins()
关于加载与卸载有以下几点声明：

load 函数使用重载的方式实现，同时支持无参加载和指定动态库路径加载，提供可选参数配置 PythonLogger 的打印等级，如果不配置，会将 PYLOG 重置为 warn 打印等级；
load() 函数进行了 Python 相关的准备工作，在进行 Python 互操作前必须调用，其中动态库查询方式请见：动态库的加载策略；
load(path: String) 函数需要用户配置动态库路径 path， path 指定到动态库文件（如：/usr/lib/libpython3.9.so），不可以配置为目录或者非动态库文件；
load 函数失败时会抛出 PythonException 异常，如果程序仍然需要继续执行，请注意 try-catch ；
unload 函数在进行完 Python 互操作时调用，否则会造成相关资源泄露；
加载和卸载操作仅需要调用一次，并且一一对应，多次调用仅第一次生效；
isload() 函数用于判断 Python 库是否被加载。
使用示例：

load 与 unload ：

import std.ffi.python.*

main(): Int64 {
    Python.load()
    Python.unload()
    Python.load("/usr/lib/libpython3.9.so")
    Python.unload()
    return 0
}
isLoad 函数:

import std.ffi.python.*

main(): Int64 {
    print("${Python.isLoad()}\n")       // false
    Python.load()
    print("${Python.isLoad()}\n")       // true
    Python.unload()
    return 0
}
动态库的加载策略
Python 库需要依赖 Python 的官方动态链接库： libpython3.x.so ，推荐版本：3.9.2，支持读取 Python3.0 以上版本。

从 Python 源码编译获取动态库：

# 在Python源码路径下：
./configure --enable-shared --with-system-ffi --prefix=/usr
make
make install
Python 的动态库按照以下方式进行自动查找：

1、使用指定的环境变量：

export PYTHON_DYNLIB=".../libpython3.9.so"
2、如果环境变量未指定，从可执行文件的依赖中查找：

需要保证可执行文件 python3 可正常执行（所在路径已添加值 PATH 环境变量中），通过对 python3 可执行文件的动态库依赖进行查询。
非动态库依赖的 Python 可执行文件无法使用（源码编译未使用 --enable-shared 编译的 Python 可执行文件，不会对动态库依赖）。
$ ldd $(which python3)
    ...
    libpython3.9d.so.1.0 => /usr/local/lib/libpython3.9d.so.1.0 (0x00007f499102f000)
    ...
3、如果无法找到可执行文件依赖，尝试从系统默认动态库查询路径中查找：

["/lib", "/usr/lib", "/usr/local/lib"]
所在路径下查询的动态库名称必须满足 libpythonX.Y.so 的命名方式，其中 X Y 分别为主版本号以及次版本号，并且支持的后缀有：d.so，m.so，dm.so，.so，支持的版本高于 python3.0，低于或等于 python3.10。如：

libpython3.9.so
libpython3.9d.so
libpython3.9m.so
libpython3.9dm.so
使用示例：

import std.ffi.python.*
import std.log.*

main(): Int64 {
    Python.load(loglevel: LogLevel.INFO)
    print("${Python.getVersion()}\n")
    Python.unload()
    return 0
}
可以开启 Python 的 INFO 级打印，查看 Python 库路径的搜索过程：

# Specifying .so by Using Environment Variables
$ export PYTHON_DYNLIB=/root/code/python_source_code/Python-3.9.2/libpython3.9d.so
$ cjc ./main.cj -o ./main && ./main
INFO: Try to get libpython path.
INFO: Found PYTHON_DYNLIB value: /root/code/python_source_code/Python-3.9.2/libpython3.9d.so
...

# Find dynamic libraries by executable file dependency.
INFO: Try to get libpython path.
INFO: Can't get path from environment PYTHON_DYNLIB, try to find it from executable file path.
INFO: Exec cmd: "ldd $(which python3)":
INFO:   ...
        libpython3.9d.so.1.0 => /usr/local/lib/libpython3.9d.so.1.0 (0x00007fbbb5014000)
        ...

INFO: Found lib: /usr/local/lib/libpython3.9d.so.1.0.
INFO: Found exec dependency: /usr/local/lib/libpython3.9d.so.1.0
...

# Search for the dynamic library in the system path.
$ unset PYTHON_DYNLIB
$ cjc ./main.cj -o ./main && ./main
INFO: Can't get path from environment PYTHON_DYNLIB, try to find it from executable file path.
INFO: Can't get path from executable file path, try to find it from system lib path.
INFO: Find in /lib.
INFO: Found lib: /lib/libpython3.9.so.
...

# Failed to find the dynamic library.
$ cjc ./main.cj -o ./main && ./main
INFO: Can't get path from environment PYTHON_DYNLIB, try to find it from executable file path.
INFO: Can't get path from executable file path, try to find it from system lib path.
INFO: Find in /lib.
INFO: Can't find lib in /lib.
INFO: Find in /usr/lib.
INFO: Can't find lib in /usr/lib.
INFO: Find in /usr/local/lib.
INFO: Can't find lib in /usr/local/lib.
An exception has occurred:
PythonException: Can't get path from system lib path, load exit.
         at std/ffi/python.std/ffi/python::(PythonException::)init(std/core::String)(stdlib/std/ffi/python/Python.cj:82)
         at std/ffi/python.std/ffi/python::(PythonBuiltins::)load(std/log::LogLevel)(stdlib/std/ffi/python/Python.cj:127)
         at default.default::main()(/root/code/debug/src/main.cj:5)
getVersion() 函数
函数原型：

public func getVersion(): Version
接口描述：

getVersion() 函数用于获取当前使用的 Python 版本。
入参返回值：

getVersion() 函数无参数，返回 Version 类对象。
异常情况：

getVersion() 函数需要保证 load 函数已被调用，否则返回的版本信息号为 0.0.0。
使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()
    var version = Python.getVersion()
    print("${version}")
    var tuple_version = version.getVersion()
    Python.unload()
    return 0
}
Import() 函数
函数原型：

public func Import(module: String): PyModule
入参返回值：

Import 函数接受一个 String 类型入参，即模块名，并且返回一个 PyModule 类型的对象。
异常情况：

Import 函数需要保证 load 函数已被调用，否则返回的 PyModule 类型对象不可用（ isAvaliable() 为 false ）；
如果找不到对应的模块，仅会报错，且返回的 PyModule 类型对象不可用（ isAvaliable() 为 false ）。
使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()
    var sys = Python.Import("sys")
    if (sys.isAvailable()) {
        print("Import sys success\n")
    }
    // Import the test.py file in the current folder.
    var test = Python.Import("test")
    if (test.isAvailable()) {
        print("Import test success\n")
    }
    var xxxx = Python.Import("xxxx")
    if (!xxxx.isAvailable()) {
        print("Import test failed\n")
    }
    Python.unload()
    return 0
}
执行结果：

Import sys success
Import test success
Import test failed
Eval() 函数
函数原型：

public func Eval(cmd: String, module!: String = "__main__"): PyObj
接口描述：

Eval() 函数用于创建一个 Python 数据类型。
入参返回值：

Eval() 接受一个 String 类型的命令 cmd ，并返回该指令的结果的 PyObj 形式；
Eval() 接受一个 String 类型的指定域，默认域为 "__main__"。
异常情况：

Eval() 接口需要保证 load 函数已被调用，否则返回的 PyObj 类型对象不可用（ isAvaliable() 为 false ）；
Eval() 如果接收的命令执行失败，Python 侧会进行报错，并且返回的 PyObj 类型对象不可用（ isAvaliable() 为 false ）。
使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()
    var a = Python.Eval("123")
    if (a.isAvailable()) {
        Python["print"]([a])
    }
    var b = Python.Eval("x = 123") // The expression in `Eval` needs have a return value.
    if (!b.isAvailable()) {
        print("b is unavailable.\n")
    }
    Python.unload()
    return 0
}
执行结果：

123
b is unavailable.
index [] 运算符重载
接口描述：

[] 函数提供了其他 Python 的内置函数调用能力。
入参返回值：

[] 函数入参接受 String 类型的内建函数名，返回类型为 PyObj 。
异常处理：

[] 函数需要保证 load 函数已被调用，否则返回的 PyObj 类型对象不可用（ isAvaliable() 为 false ）；
如果指定的函数名未找到，则会报错，且返回的 PyObj 类型对象不可用（ isAvaliable() 为 false ）。
使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()
    if (Python["type"].isAvailable()) {
        print("find type\n")
    }
    if (!Python["type1"].isAvailable()) {
        print("cant find type1\n")
    }
    Python.unload()
    return 0
}
执行结果：

find type
WARN: Dict key "type1" not found!
cant find type1
类型映射
由于 Python 与仓颉互操作基于 C API 开发，Python 与 C 的数据类型映射统一通过 PyObject 结构体指针完成，并且具有针对不同数据类型的一系列接口。对比 C 语言，仓颉具有面向对象的编程优势，因此将 PyObject 结构体指针统一封装为父类，并且被不同的数据类型进行继承。

类型映射表
仓颉类型到 Python 类型映射：

Cangjie Type	Python Type
Bool	PyBool
UInt8/Int8/Int16/UInt16/Int32/UInt32/Int64/UInt64	PyLong
Float32/Float64	PyFloat
Rune/String	PyString
Array< PyObj >	PyTuple
Array	PyList
HashMap	PyDict
HashSet	PySet
Python 类型到仓颉类型映射：

Python Type	Cangjie Type
PyBool	Bool
PyLong	Int64/UInt64
PyFloat	Float64
PyString	String
PyTuple	-
PyList	Array
PyDict	HashMap
PySet	HashSet
Python FFI 库泛型约束的接口 PyFFIType
public interface PyFFIType { }
由于部分类引入了泛型，为了对用户在泛型使用过程中进行约束，引入了抽象接口 PyFFIType；
该接口无抽象成员函数，其仅被 PyObj 和 CjObj 实现或继承，该接口不允许在包外进行实现，如果用户自定义类并实现改接口，可能发生未定义行为。
PyObj 类
与 Python 库中的结构体 PyObject 对应，对外提供细分数据类型通用的接口，如成员变量访问、函数访问、到仓颉字符串转换等。

类原型：

public open class PyObj <: ToString & PyFFIType {
    public func isAvailable(): Bool { ... }
    public open operator func [](key: String): PyObj { ... }
    public open operator func [](key: String, value!: PyObj): Unit { ... }
    public operator func ()(): PyObj { ... }
    public operator func ()(kargs: HashMap<String, PyObj>): PyObj { ... }
    public operator func ()(args: Array<PyObj>): PyObj { ... }
    public operator func ()(args: Array<PyObj>, kargs: HashMap<String, PyObj>): PyObj { ... }
    public operator func ()(args: Array<CjObj>): PyObj { ... }
    public operator func ()(args: Array<CjObj>, kargs: HashMap<String, PyObj>): PyObj { ... }
    public operator func +(b: PyObj): PyObj { ... }
    public operator func -(b: PyObj): PyObj { ... }
    public operator func *(b: PyObj): PyObj { ... }
    public operator func /(b: PyObj): PyObj { ... }
    public operator func **(b: PyObj): PyObj { ... }
    public operator func %(b: PyObj): PyObj { ... }
    public open func toString(): String { ... }
    public func hashCode(): Int64 { ... }
    public operator func ==(right: PyObj): Bool { ... }
    public operator func !=(right: PyObj): Bool { ... }
}
关于 PyObj 类的几点说明：

PyObj 不对外提供创建的构造函数，该类不能在包外进行继承，如果用户自定义类并实现改接口，可能发生未定义行为；

public func isAvailable(): Bool { ... } ：

isAvailable 接口用于判断该 PyObj 是否可用（即封装的 C 指针是否为 NULL）。
public open operator func [](key: String): PyObj { ... } ：

[](key) 用于访问 Python 类的成员或者模块中的成员等；
如果 PyObj 本身不可用（ isAvaliable() 为 false ），将抛出异常；
如果 PyObj 中不存在对应的 key ，此时由 Python 侧打印对应的错误，并返回不可用的 PyObj 类对象（ isAvaliable() 为 false ）。
public open operator func [](key: String, value!: PyObj): Unit { ... } ：

[](key, value) 设置 Python 类、模块的成员变量值为 value ；
如果 PyObj 本身不可用（ isAvaliable() 为 false ），将抛出异常；
如果 PyObj 中不存在对应的 key ，此时由 Python 侧打印对应的错误；
如果 value 值为一个不可用的对象（ isAvaliable() 为 false ），此时会将对应的 key 从模块或类中删除。
() 括号运算符重载，可调用对象的函数调用：

如果 PyObj 本身不可用（ isAvaliable() 为 false ），将抛出异常；
如果 PyObj 本身为不可调用对象，将由 Python 侧报错，且返回不可用的 PyObj 类对象（ isAvaliable() 为 false ）；
() 接受无参的函数调用；
([...]) 接受大于等于 1 个参数传递，参数类型支持仓颉类型 CjObj 和 Python 数据类型 PyObj ，需要注意的是，多个参数传递时，CjObj 和 PyObj 不可混用；
如果参数中包含不可用对象（ isAvaliable() 为 false ），此时将会抛出异常，避免发生在 Python 侧出现不可预测的程序崩溃；
() 运算符支持 kargs ，即对应 Python 的可变命名参数设计，其通过一个 HashMap 进行传递，其 key 类型 String 配置为变量名， value 类型为 PyObj 配置为参数值。
二元运算符重载：

+ 两变量相加：

基础数据类型：PyString 与 PyBool/PyLong/PyFloat 不支持相加，其他类型均可相互相加；
高级数据类型：PyDict/PySet 与所有类型均不支持相加，PyTuple/PyList 仅能与自身相加。
- 两变量相减：

基础数据类型：PyString 与 PyBool/PyLong/PyFloat/PyString 不支持相减，其他类型均可相互相减；
高级数据类型：PyDict/PySet/PyTuple/PyList 与所有类型均不支持相减。
* 两变量相乘：

基础数据类型：PyString 与 PyFloat/PyString 不支持相乘，其他类型均可相乘；
高级数据类型：PyDict/PySet 与所有类型均不支持相乘，PyTuple/PyList 仅能与 PyLong/PyBool 相乘。
/ 两变量相除：

基础数据类型：PyString 与 PyBool/PyLong/PyFloat/PyString 不支持相除，其他类型均可相互相除；如果除数为 0（False 在 Python 侧解释为 0，不可作为除数），会在 Python 侧进行错误打印；
高级数据类型：PyDict/PySet/PyTuple/PyList 与所有类型均不支持相除。
** 指数运算：

基础数据类型：PyString 与 PyBool/PyLong/PyFloat/PyString 不支持指数运算，其他类型均可进行指数运算；
高级数据类型：PyDict/PySet/PyTuple/PyList 与所有类型均不支持指数运算。
% 取余：

基础数据类型：PyString 与 PyBool/PyLong/PyFloat/PyString 不支持取余运算，其他类型均可进行取余运算；如果除数为 0（False 在 Python 侧解释为 0，不可作为除数），会在 Python 侧进行错误打印；
高级数据类型：PyDict/PySet/PyTuple/PyList 与所有类型均不支持取余运算。
以上所有错误情况均会进行 warn 级别打印，并且返回的 PyObj 不可用（isAvaliable() 为 false）。

public open func toString(): String { ... } ：

toString 函数可以将 Python 数据类型以字符串形式返回，基础数据类型将以 Python 风格返回；
如果 PyObj 本身不可用（ isAvaliable() 为 false ），将抛出异常。
hashCode 函数为封装的 Python hash 算法，其返回一个 Int64 的哈希值；

== 操作符用于判定两个 PyObj 对象是否相同，!= 与之相反，如果接口比较失败，== 返回为 false 并捕获 Python 侧报错，如果被比较的两个对象存在不可用，会抛出异常。

使用示例：

test01.py 文件：

a = 10
def function():
    print("a is", a)
def function02(b, c = 1):
    print("function02 call.")
    print("b is", b)
    print("c is", c)
同级目录下的仓颉文件 main.cj：

import std.ffi.python.*
import std.collection.*

main(): Int64 {
    Python.load()

    // Create an unavailable value.
    var a = Python.Eval("a = 10")   // SyntaxError: invalid syntax
    print("${a.isAvailable()}\n")   // false

    // Uncallable value `b` be invoked
    var b = Python.Eval("10")
    b()                           // TypeError: 'int' object is not callable

    // Import .py file.
    var test = Python.Import("test01")

    // `get []` get value of `a`.
    var p_a = test["a"]
    print("${p_a}\n")               // 10

    // `set []` set the value of a to 20.
    test["a"] = Python.Eval("20")
    test["function"]()            // a is 20

    // Call function02 with a named argument.
    test["function02"]([1], HashMap<String, PyObj>([("c", 2.toPyObj())]))

    // Set `a` in test01 to an unavailable value, and `a` will be deleted.
    test["a"] = a
    test["function"]()            // NameError: name 'a' is not defined

    Python.unload()
    0
}
CjObj 接口
接口原型及类型扩展：

public interface CjObj <: PyFFIType {
    func toPyObj(): PyObj
}
extend Bool <: CjObj {
    public func toPyObj(): PyBool { ... }
}
extend Rune <: CjObj {
    public func toPyObj(): PyString { ... }
}
extend Int8 <: CjObj {
    public func toPyObj(): PyLong { ... }
}
extend UInt8 <: CjObj {
    public func toPyObj(): PyLong { ... }
}
extend Int16 <: CjObj {
    public func toPyObj(): PyLong { ... }
}
extend UInt16 <: CjObj {
    public func toPyObj(): PyLong { ... }
}
extend Int32 <: CjObj {
    public func toPyObj(): PyLong { ... }
}
extend UInt32 <: CjObj {
    public func toPyObj(): PyLong { ... }
}
extend Int64 <: CjObj {
    public func toPyObj(): PyLong { ... }
}
extend UInt64 <: CjObj  {
    public func toPyObj(): PyLong { ... }
}
extend Float32 <: CjObj  {
    public func toPyObj(): PyFloat { ... }
}
extend Float64 <: CjObj  {
    public func toPyObj(): PyFloat { ... }
}
extend String <: CjObj  {
    public func toPyObj(): PyString { ... }
}
extend<T> Array<T> <: CjObj where T <: PyFFIType {
    public func toPyObj(): PyList<T> { ... }
}
extend<K, V> HashMap<K, V> <: CjObj where K <: Hashable & Equatable<K> & PyFFIType {
    public func toPyObj(): PyDict<K, V> { ... }
}
extend<T> HashSet<T> <: CjObj where T <: Hashable, T <: Equatable<T> & PyFFIType {
    public func toPyObj(): PySet<T> { ... }
}
关于 CjObj 类的说明：

CjObj 接口被所有基础数据类型实现并完成 toPyObj 扩展，分别支持转换为与之对应的 Python 数据类型。

PyBool 与 Bool 的映射
类原型：

public class PyBool <: PyObj {
    public init(bool: Bool) { ... }
    public func toCjObj(): Bool { ... }
}
关于 PyBool 类的几点说明

PyBool 类继承自 PyObj 类， PyBool 具有所有父类拥有的接口；
PyBool 仅允许用户使用仓颉的 Bool 类型进行构造；
toCjObj 接口将 PyBool 转换为仓颉数据类型 Bool 。
使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()

    // Creation of `PyBool`.
    var a = PyBool(true)        // The type of `a` is `PyBool`.
    var b = Python.Eval("True") // The type of `b` is `PyObj` and needs to be matched to `PyBool`.
    var c = true.toPyObj()      // The type of `c` is `PyBool`, which is the same as `a`.

    print("${a}\n")
    if (a.toCjObj()) {
        print("success\n")
    }

    if (b is PyBool) {
        print("b is PyBool\n")
    }
    Python.unload()
    0
}
执行结果：

True
success
b is PyBool
PyLong 与整型的映射
类原型：

public class PyLong <: PyObj {
    public init(value: Int64) { ... }
    public init(value: UInt64) { ... }
    public init(value: Int32) { ... }
    public init(value: UInt32) { ... }
    public init(value: Int16) { ... }
    public init(value: UInt16) { ... }
    public init(value: Int8) { ... }
    public init(value: UInt8) { ... }
    public func toCjObj(): Int64 { ... }
    public func toInt64(): Int64 { ... }
    public func toUInt64(): UInt64 { ... }
}
关于 PyLong 类的几点说明

PyLong 类继承自 PyObj 类， PyLong 具有所有父类拥有的接口；

PyLong 支持来自所有仓颉整数类型的入参构造；

toCjObj 与 toInt64 接口将 PyLong 转换为 Int64 类型；

toUInt64 接口将 PyLong 转换为 UInt64 类型；

PyLong 类型向仓颉类型转换统一转换为 8 字节类型，不支持转换为更低字节类型；

溢出问题：

toInt64 原数值（以 UInt64 赋值，赋值不报错）超出 Int64 范围判定为溢出；
toUInt64 原数值（以 Int64 赋值，赋值不报错）超出 UInt64 范围判定为溢出；
PyLong 暂不支持大数处理。

使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()

    // Creation of `PyLong`.
    var a = PyLong(10)          // The type of `a` is `PyLong`.
    var b = Python.Eval("10")   // The type of `b` is `PyObj` and needs to be matched to `PyLong`.
    var c = 10.toPyObj()        // The type of `c` is `PyLong`, which is the same as `a`.

    print("${a}\n")
    if (a.toCjObj() == 10 && a.toUInt64() == 10) {
        print("success\n")
    }

    if (b is PyLong) {
        print("b is PyLong\n")
    }
    Python.unload()
    0
}
执行结果：

10
success
b is PyLong
PyFloat 与浮点的映射
类原型：

public class PyFloat <: PyObj {
    public init(value: Float32) { ... }
    public init(value: Float64) { ... }
    public func toCjObj(): Float64 { ... }
}
关于 PyFloat 类的几点说明

PyFloat 类继承自 PyObj 类， PyFloat 具有所有父类拥有的接口；
PyBool 支持使用仓颉 Float32/Float64 类型的数据进行构造；
toCjObj 接口为了保证精度，将 PyFloat 转换为仓颉数据类型 Float64 。
使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()

    // Creation of `PyLong`.
    var a = PyFloat(3.14)       // The type of `a` is `PyFloat`.
    var b = Python.Eval("3.14") // The type of `b` is `PyObj` and needs to be matched to `PyFloat`.
    var c = 3.14.toPyObj()      // The type of `c` is `PyFloat`, which is the same as `a`.

    print("${a}\n")
    if (a.toCjObj() == 3.14) {
        print("success\n")
    }

    if (b is PyFloat) {
        print("b is PyFloat\n")
    }
    Python.unload()
    0
}
执行结果：

3.14
success
b is PyFloat
PyString 与字符、字符串的映射
类原型：

public class PyString <: PyObj {
    public init(value: String) { ... }
    public init(value: Rune) { ... }
    public func toCjObj(): String { ... }
    public override func toString(): String { ... }
}
关于 PyString 类的几点说明

PyString 类继承自 PyObj 类， PyString 具有所有父类拥有的接口；
PyString 支持使用仓颉 Rune/String 类型的数据进行构造；
toCjObj/toString 接口为将 PyString 转换为仓颉数据类型 String 。
使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()

    // Creation of `PyString`.
    var a = PyString("hello python")        // The type of `a` is `PyString`.
    var b = Python.Eval("\"hello python\"") // The type of `b` is `PyObj` and needs to be matched to `PyString`.
    var c = "hello python".toPyObj()        // The type of `c` is `PyString`, which is the same as `a`.

    print("${a}\n")
    if (a.toCjObj() == "hello python") {
        print("success\n")
    }

    if (b is PyString) {
        print("b is PyString\n")
    }
    Python.unload()
    0
}
执行结果：

hello python
success
b is PyString
PyTuple 类型
类原型：

public class PyTuple <: PyObj {
    public init(args: Array<PyObj>) { ... }
    public operator func [](key: Int64): PyObj { ... }
    public func size(): Int64 { ... }
    public func slice(begin: Int64, end: Int64): PyTuple { ... }
}
关于 PyTuple 类的几点说明

PyTuple 与 Python 中的元组类型一致，即 Python 代码中使用 (...) 的变量；
PyTuple 类继承自 PyObj 类， PyTuple 具有所有父类拥有的接口；
PyTuple 支持使用仓颉 Array 来进行构造， Array 的元素类型必须为 PyObj （Python 不同数据类型均可以使用 PyObj 传递，即兼容 Tuple 中不同元素的不同数据类型），当成员中包含不可用对象时，会抛出异常；
[] 操作符重载：
父类 PyObj 中 [] 入参类型为 String 类型，该类对象调用时能够访问或设置 Python 元组类型内部成员变量或者函数；
子类 PyTuple 支持使用 [] 对元素进行访问，如果角标 key 超出 [0, size()) 区间，会进行报错，并且返回不可用的 PyObj 对象；
由于 Python 的元组为不可变对象，未进行 set [] 操作符重载。
size 函数用于获取 PyTuple 的长度；
slice 函数用于对源 PyTuple 进行剪裁，并返回一个新的 PyTuple , 如果 slice 的入参 begin 和 end 不在 [0, size()) 区间内，仍会正常裁切。
使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()

    // Creation of `PyTuple`.
    var a = PyTuple(["Array".toPyObj(), 'a'.toPyObj(), 1.toPyObj(), 1.1.toPyObj()])
    var b = match (Python.Eval("('Array', 'a', 1, 1.1)")) {
        case val: PyTuple => val
        case _ => throw PythonException()
    }

    // Usage of size
    println(a.size())           // 4

    // Usage of slice
    println(a.slice(1, 2))      // ('a',). This print is same as Python code `a[1: 2]`.
    println(a.slice(-1, 20))    // ('Array', 'a', 'set index 3 to String', 1.1)

    Python.unload()
    return 0
}
执行结果：

4
('a',)
('Array', 'a', 1, 1.1)
PyList 与 Array 的映射
类原型：

public class PyList<T> <: PyObj where T <: PyFFIType {
    public init(args: Array<T>) { ... }
    public operator func [](key: Int64): PyObj { ... }
    public operator func [](key: Int64, value!: T): Unit { ... }
    public func toCjObj(): Array<PyObj> { ... }
    public func size(): Int64 { ... }
    public func insert(index: Int64, value: T): Unit { ... }
    public func append(item: T): Unit { ... }
    public func slice(begin: Int64, end: Int64): PyList<T> { ... }
}
关于 PyList 类的几点说明

PyList 类与 Python 中的列表类型一致，即 Python 代码中使用 [...] 的变量；

PyList 类继承自 PyObj 类， PyList 具有所有父类拥有的接口，该类由于对仓颉的 Array 进行映射，因此该类引入了泛型 T ， T 类型约束为 PyFFIType 接口的子类；

PyList 类可以通过仓颉的 Array 类型进行构造， Array 的成员类型同样约束为 PyFFIType 接口的子类；

[] 操作符重载：

父类 PyObj 中 [] 入参类型为 String 类型，该类对象调用时仅能访问或设置 Python 内部成员变量或者函数；
该类中的 [] 入参类型为 Int64 ，即对应 Array 的角标值，其范围为 [0, size())，如果入参不在范围内，将进行报错，并且返回的对象为不可用；
[] 同样支持 get 以及 set ，并且 set 时， value 类型为 T ，如果 value 其中包含不可用的 Python 对象时，会抛出异常。
toCjObj 函数支持将 PyList 转换为仓颉的 Array<PyObj>，请注意，此时并不会转换为 Array<T>；

size 函数返回 PyList 的长度；

insert 函数将在 index 位置插入 value ，其后元素往后移，index 不在 [0, size()) 可以正常插入，如果 value 为不可用对象，将会抛出异常；

append 函数将 item 追加在 PyList 最后，如果 value 为不可用对象，将会抛出异常；

slice 函数用于截取 [begin, end) 区间内的数据并且返回一个新的 PyList , begin 和 end 不在 [0, size()) 也可以正常截取。

使用示例：

import std.ffi.python.*

main(): Int64 {
    Python.load()

    // Creation of `PyList`.
    var a = PyList<Int64>([1, 2, 3])
    var b = match (Python.Eval("[1, 2, 3]")) {
        case val: PyList<PyObj> => val
        case _ => throw PythonException()
    }
    var c = [1, 2, 3].toPyObj()

    // Usage of `[]`
    println(a["__add__"]([b]))   // [1, 2, 3, 1, 2, 3]
    a[1]
    b[1]
    a[2] = 13
    b[2] = 15.toPyObj()

    // Usage of `toCjObj`
    var cjArr = a.toCjObj()
    for (v in cjArr) {
        print("${v} ")          // 1 2 13
    }
    print("\n")

    // Usage of `size`
    println(a.size())           // 3

    // Usage of `insert`
    a.insert(1, 4)              // [1, 4, 2, 13]
    a.insert(-100, 5)           // [5, 1, 4, 2, 13]
    a.insert(100, 6)            // [5, 1, 4, 2, 13, 6]
    b.insert(1, 4.toPyObj())    // [1, 4, 2, 15]

    // Usage of `append`
    a.append(7)                 // [5, 1, 4, 2, 13, 6, 7]
    b.append(5.toPyObj())       // [1, 4, 2, 15, 5]

    // Usage of `slice`
    a.slice(1, 2)               // [1]
    a.slice(-100, 100)          // [5, 1, 4, 2, 13, 6, 7]
    b.slice(-100, 100)          // [1, 4, 2, 15, 5]

    return 0
}
执行结果：

[1, 2, 3, 1, 2, 3]
1 2 13
3
PyDict 与 HashMap 的映射
类原型：

public class PyDict<K, V> <: PyObj where K <: Hashable & Equatable<K> & PyFFIType {
    public init(args: HashMap<K, V>) { ... }
    public func getItem(key: K): PyObj { ... }
    public func setItem(key: K, value: V): Unit { ... }
    public func toCjObj(): HashMap<PyObj, PyObj> { ... }
    public func contains(key: K): Bool { ... }
    public func copy(): PyDict<K, V> { ... }
    public func del(key: K): Unit { ... }
    public func size(): Int64 { ... }
    public func empty(): Unit { ... }
    public func items(): PyList<PyObj> { ... }
    public func values(): PyList<PyObj> { ... }
    public func keys(): PyList<PyObj> { ... }
}
关于 PyDict 类的几点说明

PyDict 与 Python 的字典类型一致，即 Python 代码中使用 { a: b } 的变量；

PyDict 类继承自 PyObj 类， PyDict 具有所有父类拥有的接口，该类由于对仓颉的 HashMap 进行映射，因此该类引入了泛型 <K, V> ，其中 K 类型约束为 PyFFIType 接口的子类，且可被 Hash 计算以及重载了 == 与 != 运算符；

PyDict 接受来自仓颉类型 HashMap 的数据进行构造：

K 仅接受 CjObj 或 PyObj 类型或其子类；
相同的 Python 数据其值也相同，例如 Python.Eval("1") 与 1.toPyObj() 为 == 关系。
getItem 函数用于获取 PyDict 对应键值的 value ，如果键值无法找到，会进行报错并返回不可用的 PyObj ，如果配置的值 key 或为 value 为 PyObj 类型且不可用，此时抛出异常；

setItem 函数用于配置 PyDict 对应键值的 value ，如果对应键值无法找到，会进行插入，如果配置的值 key 或为 value 为 PyObj 类型且不可用，此时抛出异常；

toCjObj 函数用于将 PyDict 转换为 HashMap<PyObj, PyObj> 类型；

contains 函数用于判断 key 值是否包含在当前字典中，返回类型为 Bool 型，如果接口失败，进行报错，并且返回 false；

copy 函数用于拷贝当前字典，并返回一个新的 PyDict<T> 类型，如果拷贝失败，返回的 PyDict 不可用；

del 函数用于删除对应 key 的值，如果 key 值为 PyObj 类型且不可用，会抛出异常；

size 函数用于返回当前字典的长度；

empty 函数用于清空当前字典内容；

items 函数用于获取一个 Python list 类型的键值对列表，可以被迭代访问；

values 函数用于获取一个 Python list 类型的值列表，可以被迭代访问；

keys 函数用于获取一个 Python list 类型的键列表，可以被迭代访问。

使用示例：

import std.ffi.python.*
import std.collection.*

main() {
    Python.load()

    // Creation of `PyDict`
    var a = PyDict(HashMap<Int64, Int64>([(1, 1), (2, 2)]))             // The key type is `CjObj`.
    var b = PyDict(HashMap<PyObj, Int64>([(Python.Eval("1"), 1), (Python.Eval("2"), 2)]))   // The key type is `PyObj`.
    var c = match (Python.Eval("{'pydict': 1, 'hashmap': 2, 3: 3, 3.1: 4}")) {
        case val: PyDict<PyObj, PyObj> => val       // Python side return `PyDict<PyObj, PyObj>`
        case _ => throw PythonException()
    }
    var d = HashMap<Int64, Int64>([(1, 1), (2, 2)]).toPyObj()

    // Usage of `getItem`
    println(a.getItem(1))               // 1
    println(b.getItem(1.toPyObj()))     // 1

    // Usage of `setItem`
    a.setItem(1, 10)
    b.setItem(1.toPyObj(), 10)
    println(a.getItem(1))               // 10
    println(b.getItem(1.toPyObj()))     // 10

    // Usage of `toCjObj`
    var hashA = a.toCjObj()
    for ((k, v) in hashA) {
        print("${k}: ${v}, ")           // 1: 10, 2: 2,
    }
    print("\n")
    var hashB = b.toCjObj()
    for ((k, v) in hashB) {
        print("${k}: ${v}, ")           // 1: 10, 2: 2,
    }
    print("\n")

    // Usage of `contains`
    println(a.contains(1))              // true
    println(a.contains(3))              // false
    println(b.contains(1.toPyObj()))    // true

    // Usage of `copy`
    println(a.copy())                   // {1: 10, 2: 2}

    // Usage of `del`
    a.del(1)                            // Delete the key-value pair (1: 1).

    // Usage of `size`
    println(a.size())                   // 1

    // Usage of `empty`
    a.empty()                           // Clear all elements in dict.

    // Usage of `items`
    for (i in b.items()) {
        print("${i} ")                  // (1, 10) (2, 2)
    }
    print("\n")

    // Usage of `values`
    for (i in b.values()) {
        print("${i} ")                  // 10 2
    }
    print("\n")

    // Usage of `keys`
    for (i in b.keys()) {
        print("${i} ")                  // 1, 2
    }
    print("\n")

    Python.unload()
}
PySet 与 HashSet 的映射
类原型：

public class PySet<T> <: PyObj where T <: Hashable, T <: Equatable<T> & PyFFIType {
    public init(args: HashSet<T>) { ... }
    public func toCjObj(): HashSet<PyObj> { ... }
    public func contains(key: T): Bool { ... }
    public func add(key: T): Unit { ... }
    public func pop(): PyObj { ... }
    public func del(key: T): Unit { ... }
    public func size(): Int64 { ... }
    public func empty(): Unit { ... }
}
关于 PySet 类的几点说明

PySet 对应的是 Python 中的集合的数据类型，当元素插入时会使用 Python 内部的 hash 算法对集合元素进行排序（并不一定按照严格升序，一些方法可能因此每次运行结果不一致）。

PySet 类继承自 PyObj 类， PySet 具有所有父类拥有的接口，该类由于对仓颉的 HashSet 进行映射，因此该类引入了泛型 T ， T 类型约束为 PyFFIType 接口的子类，且可被 Hash 计算以及重载了 == 与 != 运算符；

PySet 接受来自仓颉类型 HashMap 的数据进行构造：

K 仅接受 CjObj 或 PyObj 类型或其子类；
相同的 Python 数据其值也相同，例如 Python.Eval("1") 与 1.toPyObj() 为 == 关系。
toCjObj 函数用于将 PySet<T> 转为 HashSet<PyObj> 需要注意的是此处只能转为元素类型为 PyObj 类型；

contains 函数用于判断 key 是否在当前字典中存在， key 类型为 T ；

add 函数可以进行值插入，当 PySet 中已存在键值，则插入不生效，如果 key 为 PyObj 且不可用，则会抛出异常；

pop 函数将 PySet 中的第一个元素取出；

del 删除对应的键值，如果 key 不在 PySet 中，则会报错并正常退出，如果 key 为 PyObj 且不可用，则会抛出异常；

size 用于返回 PySet 的长度；

empty 用于清空当前 PySet 。

注意：

调用 toCjObj 完后，所有元素将被 pop 出来，此时原 PySet 将会为空（ size 为 0，原 PySet 仍然可用）；

使用示例：

import std.ffi.python.*
import std.collection.*

main() {
    Python.load()

    // Creation of `PySet`
    var a = PySet<Int64>(HashSet<Int64>([1, 2, 3]))
    var b = match (Python.Eval("{'PySet', 'HashSet', 1, 1.1, True}")) {
        case val: PySet<PyObj> => val
        case _ => throw PythonException()
    }
    var c = HashSet<Int64>([1, 2, 3]).toPyObj()

    // Usage of `toCjObj`
    var cja = a.toCjObj()
    println(a.size())                           // 0

    // Usage of `contains`
    println(b.contains("PySet".toPyObj()))      // true

    // Usage of `add`
    a.add(2)
    println(a.size())   // 1
    a.add(2)            // Insert same value, do nothing.
    println(a.size())   // 1
    a.add(1)            // Insert `1`.

    // Usage of `pop`
    println(a.pop())    // 1. Pop the first element.
    println(a.size())   // 1

    // Usage of `del`
    c.del(2)
    println(c.contains(2))  // false

    // Usage of `empty`
    println(c.size())   // 2
    c.empty()
    println(c.size())   // 0

    Python.unload()
}
PySlice 类型
PySlice 类型与 Python 内建函数 slice() 的返回值用法一致，可以被用来标识一段区间及步长，可以用来作为可被切片的类型下标值来剪裁获取子串。为了方便从仓颉侧构造， PySlice 类可以与仓颉的 Range 区间类型进行互相转换，详细描述见以下。

类原型：

public class PySlice<T> <: PyObj where T <: Countable<T> & Comparable<T> & Equatable<T> & CjObj {
    public init(args: Range<T>) { ... }
    public func toCjObj(): Range<Int64> { ... }
}
关于 PySlice 的几点说明：

PySlice 可以使用仓颉的 Range 类型来进行构造，并且支持 Range 的语法糖，其中泛型 T 在原有 Range 约束的同时，加上约束在来自 CjObj 的实现，不支持 PyObj 类型；
toCjObj 函数支持将 PySlice 转为仓颉 Range 的接口，应注意此时 Range 的泛型类型为 Int64 类型的整型；
如果希望把 PySlice 类型传递给 PyString/PyList/PyTuple 或者是其他可被 slice 的 PyObj 类型，可以通过其成员函数 __getitem__ 进行传递，详情见示例。
使用示例：

import std.ffi.python.*

main() {
    Python.load()
    var range = 1..6:2

    // Create a PySlice.
    var slice1 = PySlice(range)
    var slice2 = match (Python["slice"]([0, 6, 2])) {
        case val: PySlice<Int64> => val
        case _ => throw PythonException()
    }
    var slice3 = range.toPyObj()

    // Use PySlice in PyString.
    var str = PyString("1234567")
    println(str["__getitem__"]([range]))    // 246
    println(str["__getitem__"]([slice1]))   // 246

    // Use PySlice in PyList.
    var list = PyList(["a", "b", "c", "d", "e", "f", "g", "h"])
    println(list["__getitem__"]([range]))   // ['b', 'd', 'f']
    println(list["__getitem__"]([slice1]))  // ['b', 'd', 'f']

    // Use PySlice in PyTuple.
    var tup = PyTuple(list.toCjObj())
    println(tup["__getitem__"]([range]))    // ('b', 'd', 'f')
    println(tup["__getitem__"]([slice1]))   // ('b', 'd', 'f')

    Python.unload()
    0
}
执行结果：

246
246
['b', 'd', 'f']
['b', 'd', 'f']
('b', 'd', 'f')
('b', 'd', 'f')
PyObj 的迭代器类型 PyObjIterator
代码原型：

PyObj 的扩展：

extend PyObj <: Iterable<PyObj> {
    public func iterator(): Iterator<PyObj> { ... }
}
PyObjIterator 类型：

public class PyObjIterator <: Iterator<PyObj> {
    public init(obj: PyObj) { ... }
    public func next(): Option<PyObj> { ... }
    public func iterator(): Iterator<PyObj> { ... }
}
关于 PyObjIterator 的几点说明：

获取 PyObjIterator 可以通过 PyObj 的 iterator 方法获取；

PyObjIterator 允许被外部构造，如果提供的 PyObj 不可以被迭代或提供的 PyObj 不可用，则会直接抛出异常；

可以被迭代的对象有：PyString/PyTuple/PyList/PySet/PyDict；
直接对 PyDict 进行迭代时，迭代的为其键 key 的值。
next 函数用于对该迭代器进行迭代；

iterator 方法用于返回本身。

使用示例：

import std.ffi.python.*
import std.collection.*

main() {
    Python.load()

    // iter of PyString
    var S = PyString("Str")
    for (s in S) {
        print("${s} ")      // S t r
    }
    print("\n")

    // iter of PyTuple
    var T = PyTuple(["T".toPyObj(), "u".toPyObj(), "p".toPyObj()])
    for (t in T) {
        print("${t} ")      // T u p
    }
    print("\n")

    // iter of PyList
    var L = PyList(["L", "i", "s", "t"])
    for (l in L) {
        print("${l} ")      // L i s t
    }
    print("\n")

    // iter of PyDict
    var D = PyDict(HashMap<Int64, String>([(1, "D"), (2, "i"), (3, "c"), (4, "t")]))
    for (d in D) {
        print("${d} ")      // 1 2 3 4, dict print keys.
    }
    print("\n")

    // iter of PySet
    var Se = PySet(HashSet<Int64>([1, 2, 3]))
    for (s in Se) {
        print("${s} ")      // 1 2 3
    }
    print("\n")
    0
}
执行结果：

S t r
T u p
L i s t
1 2 3 4
1 2 3
仓颉与 Python 的注册回调
Python 互操作库支持简单的函数注册及 Python 对仓颉函数调用。

Python 回调仓颉代码通过需要通过 C 作为介质进行调用，并且使用到了 Python 的三方库： ctypes 以及 _ctypes 。

Cangjie 类型、C 类型与 Python 类型之间的映射
基础数据对照如下表：

Cangjie Type	CType	Python Type
Bool	PyCBool	PyBool
Rune	PyCWchar	PyString
Int8	PyCByte	PyLong
UInt8	PyCUbyte/PyCChar	PyLong
Int16	PyCShort	PyLong
UInt16	PyCUshort	PyLong
Int32	PyCInt	PyLong
UInt32	PyCUint	PyLong
Int64	PyCLonglong	PyLong
UInt64	PyCUlonglong	PyLong
Float32	PyCFloat	PyFloat
Float64	PyCDouble	PyFloat
[unsupport CPointer as param] CPointer<T>	PyCPointer	ctypes.pointer
[unsupport CString as param] CString	PyCCpointer	ctypes.c_char_p
[unsupport CString as param] CString	PyCWcpointer	ctypes.c_wchar_p
Unit	PyCVoid	-
Cangjie Type 是在仓颉侧修饰的变量类型，无特殊说明则支持传递该类型参数给 Python 代码，并且支持从 Python 传递给仓颉；
PyCType 为仓颉侧对应的 PyCFunc 接口配置类型，详细见PyCFunc 类原型以及示例展示；
Python Type 是在仓颉侧的类型映射，无指针类型映射，不支持从仓颉侧调用 Python 带有指针的函数；
PyCCpointer 与 PyCWcpointer 同样都是映射到 CString ，两者区别为 PyCCpointer 为 C 中的字符串， PyCWcpointer 仅为字符指针，即使传递多个字符，也只取第一个字符；
类型不匹配将会导致不可预测的结果。
PyCFunc 类原型
PyCFunc 是基于 Python 互操作库和 Python 三方库 ctype/_ctype 的一个 PyObj 子类型，该类型可以直接传递给 Python 侧使用。 PyCFunc 为用户提供了注册仓颉的 CFunc 函数给 Python 侧，并且支持由 Python 回调 CFunc 函数的能力。

代码原型：

public enum PyCType {
    PyCBool |
    PyCChar |
    PyCWchar |
    PyCByte |
    PyCUbyte |
    PyCShort |
    PyCUshort |
    PyCInt |
    PyCUint |
    PyCLonglong |
    PyCUlonglong |
    PyCFloat |
    PyCDouble |
    PyCPointer |
    PyCCpointer |
    PyCWcpointer |
    PyCVoid
}

public class PyCFunc <: PyObj {
    public init(f: CPointer<Unit>, argsTy!: Array<PyCType> = [], retTy!: PyCType = PyCType.PyCVoid) { ... }
    public func setArgTypes(args: Array<PyCType>): PyCFunc { ... }
    public func setRetTypes(ret: PyCType): PyCFunc { ... }
}
关于类的几点说明：

PyCFunc 继承自 PyObj ，可以使用父类的部分接口（如果不支持的接口会相应报错）；

init 允许外部用户构造，必须提供函数指针作为第一个参数（仓颉侧需要将 CFunc 类型转换为 CPointer<Unit> 类型），后面两个可选参数分别为入参类型的数组、返回值类型；

这里特别声明，如果传入的指针并非函数指针会导致函数调用时程序崩溃（库层面无法进行拦截）。

setArgTypes/setRetTypes 函数用于配置参数和返回值类型，支持的参数见 PyCType 枚举；

父类中的 () 操作符，支持在仓颉侧调用该注册的 CFunc 函数；

该类可以直接传递给 Python 侧使用，也可以在仓颉侧直接调用（如果该类构造时使用非函数指针，这里调用将会崩溃）；

该类支持类似 Js 的链式调用。

示例
1、准备仓颉的 CFunc 函数：

@C
func cfoo(a: Bool, b: Int32, c: Int64): CPointer<Unit> {
    print("cfoo called.\n")
    print("${a}, ${b}, ${c}\n")
    return CPointer<Unit>()
}
2、构造 PyCFunc 类对象：

import std.ffi.python.*

// Define the @C function.
@C
func cfoo(a: Bool, b: Int32, c: Int64): CPointer<Unit> {
    print("cfoo called.\n")
    print("${a}, ${b}, ${c}\n")
    return CPointer<Unit>()
}

main() {
    Python.load()
    /*
    Construct PyCFunc class.
    Set args type:  Bool -> PyCBool
                    Int32 -> PyCInt
                    Int64 -> PyCLonglong
                    CPointer<Unit> -> PyCPointer
    */
    var f1 = PyCFunc(unsafe {CPointer<Unit>(cfoo)},
                    argsTy: [PyCBool, PyCInt, PyCLonglong],
                    retTy: PyCPointer)

    // You also can use it by chain-call.
    var f2 = PyCFunc(unsafe {CPointer<Unit>(cfoo)})
            .setArgTypes([PyCBool, PyCInt, PyCLonglong])
            .setRetTypes(PyCPointer)([true, 1, 2])

    // Call f1
    f1([true, 1, 2])
    f1([PyBool(true), PyLong(1), PyLong(2)])

    Python.unload()
    0
}
编译仓颉文件并执行：

$ cjc ./main.cj -o ./main && ./main
cfoo called.
true, 1, 2
cfoo called.
true, 1, 2
cfoo called.
true, 1, 2
3、将函数注册给 Python 并且由 Python 进行调用：

Python 代码如下：

# File test.py

# `foo` get a function pointer and call it.
def foo(func):
    func(True, 10, 40)
对上面仓颉 main 进行修改：

import std.ffi.python.*

// Define the @C function.
@C
func cfoo(a: Bool, b: Int32, c: Int64): CPointer<Unit> {
    print("cfoo called.\n")
    print("${a}, ${b}, ${c}\n")
    return CPointer<Unit>()
}

main() {
    Python.load()

    var f1 = PyCFunc(unsafe {CPointer<Unit>(cfoo)},
                    argsTy: [PyCBool, PyCInt, PyCLonglong],
                    retTy: PyCPointer)

    // Import test.py
    var cfunc01 = Python.Import("test")

    // Call `foo` and transfer `f1`
    cfunc01["foo"]([f1])

    Python.unload()
    0
}
4、Python 侧传递指针到仓颉侧：

为 Python 文件增加函数：

# File test.py

# If you want transfer pointer type to Cangjie CFunc, you need import ctypes.
import ctypes.*

# `foo` get a function pointer and call it.
def foo(func):
    func(True, 10, 40)

# `fooptr` get a function pointer and call it with pointer type args.
def fooptr(func):
    a = c_int(10)
    # c_char_p will get whole symbols, but c_wchar_p only get first one symbol 'd'.
    func(pointer(a), c_char_p(b'abc'), c_wchar_p('def'))
修改仓颉代码：

import std.ffi.python.*

var x = Python.load()

// Modify the `foo` param type to pointer.
@C
func foo(a: CPointer<Int64>, b: CString, c: CString): Unit {
    print("${unsafe {a.read(0)}}, ${b.toString()}, ${c.toString()}\n")
}

main(): Int64 {

    var f1 = PyCFunc(unsafe {CPointer<Unit>(foo)},
                    argsTy: [PyCPointer, PyCCpointer, PyCWcpointer],
                    retTy: PyCVoid)

    // Import test.py
    var test = Python.Import("test")

    // Call `fooptr` and transfer `f1`
    test["fooptr"]([f1])
    return 0
}
由于仓颉侧调用函数不能将指针类型传递给 Python 库，所以该处仅支持在 Python 侧进行调用。

对其编译并执行：

$ cjc ./main.cj -o ./main && ./main
10, abc, d