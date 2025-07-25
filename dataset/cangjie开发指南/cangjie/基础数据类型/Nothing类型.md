Nothing 类型
Nothing 是一种特殊的类型，它不包含任何值，并且 Nothing 类型是所有类型的子类型。

break、continue、return 和 throw 表达式的类型是 Nothing，程序执行到这些表达式时，它们之后的代码将不会被执行。其中 break、continue 只能在循环体中使用，return 只能在函数体中使用。

包围着的循环体“无法穿越”函数边界。在下面的例子中，break 出现在函数 f 中，外层的 while 循环体不被视作包围着它的循环体；continue 出现在 lambda 表达式中，外层的 while 循环体不被视作包围着它的循环体。

while (true) {
    func f() {
        break // Error, break must be used directly inside a loop
    }
    let g = { =>
        continue // Error, continue must be used directly inside a loop
    }
}
由于函数的形参和其默认值不属于该函数的函数体，所以下面例子中的 return 表达式缺少包围它的函数体——它既不属于外层函数 f（因为内层函数定义 g 已经开始），也不在内层函数 g 的函数体中：

func f() {
    func g(x!: Int64 = return) { // Error, return must be used inside a function body
        0
    }
    1
}
注意：

目前编译器还不允许在使用类型的地方显式地使用 Nothing 类型。