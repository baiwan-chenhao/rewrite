仓颉编程语言库 API
仓颉编程语言库包括 std 模块（标准库模块）和一些常用的扩展模块，每个模块下包含若干包，提供与该模块相关的具体而丰富的功能。

标准库为开发者提供了最通用的 API，包括输入输出功能、基础数据结构和算法、日期和时间表示等。扩展库则专注于某一领域，如 compress 模块提供压缩解压能力，crypto 模块提供加解密相关的能力，net 模块专注于提供高效的网络协议解析和网络通信能力。

标准库和官方提供的扩展库均遵守仓颉语言编程规范，在功能、性能、安全等方面符合官方标准。

说明：

标准库和官方提供的扩展库目前都随仓颉编译器、工具链一起发布，不需要用户单独下载。
根据后续演进计划，扩展库可能会从仓颉编译器、工具链发布件中剥离，放入专门的仓库管理。
使用介绍
在仓颉编程语言中，包是编译的最小单元，每个包可以单独输出 AST 文件、静态库文件、动态库文件等产物。包可以定义子包，从而构成树形结构。没有父包的包称为 root 包，root 包及其子包（包括子包的子包）构成的整棵树称为模块（module）。模块的名称与 root 包相同，是第三方开发者发布的最小单元。

包的导入规则如下：

可以导入某个包中的一个顶层声明或定义，语法如下：

import fullPackageName.itemName
其中 fullPackageName 为完整路径包名，itemName 为声明的名字，例如：

import std.collection.ArrayList
如果要导入的多个 itemName 同属于一个 fullPackageName，可以使用：

import fullPackageName.{itemName[, itemName]*}
例如：

import std.collection.{ArrayList, HashMap}
还可以将 fullPackageName 包中所有 public 修饰的顶层声明或定义全部导入，语法如下：

import fullPackageName.*
例如：

import std.collection.*`。
模块列表
当前仓颉标准库提供了如下模块：

模块名	功能
std	std 模块意指标准库，标准库是指在编程语言中预先定义的一组函数、类、结构体等，旨在提供常用的功能和工具，以便开发者能够更快速、更高效地编写程序。
compress	compress 模块提供压缩解压功能。
crypto	crypto 模块提供安全加密能力。
encoding	encoding 模块提供字符编解码功能。
fuzz	fuzz 模块提供基于覆盖率反馈的模糊测试能力。
log	log 模块提供了日志记录相关的能力。
net	net 模块提供了网络通信相关的能力。
serialization	serialization 模块提供了序列化和反序列化能力。