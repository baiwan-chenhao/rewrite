if-let 表达式
if-let 表达式首先对条件中 <- 右侧的表达式进行求值，如果此值能匹配 <- 左侧的模式，则执行 if 分支，否则执行 else 分支（可省略）。例如：

main() {
    let result = Option<Int64>.Some(2023)

    if (let Some(value) <- result) {
        println("操作成功，返回值为：${value}")
    } else {
        println("操作失败")
    }
}
运行以上程序，将输出：

操作成功，返回值为：2023
对于以上程序，如果将 result 的初始值修改为 Option<Int64>.None，则 if-let 的模式匹配会失败，将执行 else 分支：

main() {
    let result = Option<Int64>.None

    if (let Some(value) <- result) {
        println("操作成功，返回值为：${value}")
    } else {
        println("操作失败")
    }
}
运行以上程序，将输出：

操作失败