HashMap 的 get/put/contains 函数
此用例展示了 HashMap 的基本使用方法。

代码如下：

import std.collection.*
main() {
    var map: HashMap<String, Int64> = HashMap<String, Int64>()
    map.put("a", 99) // map : [("a", 99)]
    map.put("b", 100) // map : [("a", 99), ("b", 100)]
    var a = map.get("a")
    var bool = map.contains("a")
    print("a=${a.getOrThrow()} ")
    print("bool=${bool.toString()}")
    return 0
}
运行结果如下:


a=99 bool=true

HashMap 的 putAll/remove/clear 函数
此用例展示了 HashMap 的基本使用方法。

代码如下：

import std.collection.*
main() {
    var map: HashMap<String, Int64> = HashMap<String, Int64>()
    var arr: Array<(String, Int64)> = [("d", 11), ("e", 12)]
    map.putAll(arr) // map : [("d", 11), ("e", 12)]
    var d = map.get("d")
    print("d=${d.getOrThrow()} ")
    map.remove("d") // map : [("e", 12)]
    var bool = map.contains("d")
    print("bool=${bool.toString()} ")
    map.clear() // map: []
    var bool1 = map.contains("e")
    print("bool1=${bool1.toString()}")
    return 0
}
运行结果如下:


d=11 bool=false bool1=false

HashSet 的 put/iterator/remove 函数


此用例展示了 HashSet 的基本使用方法。

代码如下：

import std.collection.*
/* 测试 */
main() {
    var set: HashSet<String> = HashSet<String>() // set: []
    set.put("apple") // set: ["apple"]
    set.put("banana") // set: ["apple", "banana"], not in order
    set.put("orange") // set: ["apple", "banana", "orange"], not in order
    set.put("peach") // set: ["apple", "banana", "orange", "peach"], not in order
    var itset = set.iterator()
    while(true) {
        var value = itset.next()
        match(value) {
            case Some(v) =>
                if (!set.contains(v)) {
                    print("Operation failed")
                    return 1
                } else { println(v) }
            case None => break
        }
    }
    set.remove("apple") // set: ["banana", "orange", "peach"], not in order
    println(set)
    return 0
}
由于 Set 中的顺序不是固定的，因此运行结果可能如下：


apple
banana
orange
peach
[banana, orange, peach]