while-let 表达式
while-let 表达式首先对条件中 <- 右侧的表达式进行求值，如果此值能匹配 <- 左侧的模式，则执行循环体，然后重复执行此过程。如果模式匹配失败，则结束循环，继续执行 while-let 表达式之后的代码。例如：

import std.random.*

// 此函数模拟在通信中接收数据，获取数据可能失败
func recv(): Option<UInt8> {
    let number = Random().nextUInt8()
    if (number < 128) {
        return Some(number)
    }
    return None
}

main() {
    // 模拟循环接收通信数据，如果失败就结束循环
    while (let Some(data) <- recv()) {
        println(data)
    }
    println("receive failed")
}
运行以上程序，可能的输出为：


73
94
receive failed