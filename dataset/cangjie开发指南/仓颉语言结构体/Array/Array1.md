operator func [](Range<Int64>, T)
public operator func [](range: Range<Int64>, value!: T): Unit
功能：用指定的值对本数组一个连续范围的元素赋值。

注意：

如果参数 range 是使用 Range 构造函数构造的 Range 实例，有如下行为：
start 的值就是构造函数传入的值本身，不受构造时传入的 hasStart 的值的影响。
hasEnd 为 false 时，end 值不生效，且不受构造时传入的 isClosed 的值的影响，该数组切片取到原数组最后一个元素。
range 的步长只能为 1。
切片不会对数组数据进行拷贝，是对原数据特定区间的引用。
参数：

range: Range<Int64> - 需要修改的数组范围，range 表示的范围不能超过数组范围。
value!: T - 修改的目标值。
异常：

IllegalArgumentException - 如果 range 的步长不等于 1，抛出异常。
IndexOutOfBoundsException - 如果 range 表示的数组范围无效，抛出异常。
示例：

main() {
    let arr = [0, 1, 2, 3, 4, 5]
    arr[1..3] = 10
    println(arr)
}
运行结果：

[0, 10, 10, 3, 4, 5]
extend<T> Array<T> <: Collection<T>
extend<T> Array<T> <: Collection<T>
父类型：

Collection<T>
prop size
public prop size: Int64
功能：获取元素数量。

类型：Int64

func isEmpty()
public func isEmpty(): Bool
功能：判断数组是否为空。

返回值：

Bool - 如果数组为空，返回 true，否则，返回 false。
func iterator()
public func iterator(): Iterator<T>
功能：获取当前数组的迭代器，用于遍历数组。

返回值：

Iterator<T> - 当前数组的迭代器。
func toArray()
public func toArray(): Array<T>
功能：根据当前 Array 实例拷贝一个新的 Array 实例。

返回值：

Array<T> - 拷贝得到的新的 Array 实例。
extend<T> Array<T> <: Equatable<Array<T>> where T <: Equatable<T>
extend<T> Array<T> <: Equatable<Array<T>> where T <: Equatable<T>
功能：为 Array<T> 类型扩展 Equatable<Array<T>> 接口实现，支持判等操作。

父类型：

Equatable<Array<T>>
func contains(T)
public func contains(element: T): Bool
功能：查找当前数组是否包含指定元素。

参数：

element: T - 需要查找的目标元素。
返回值：

Bool - 如果存在，则返回 true，否则返回 false。
func indexOf(Array<T>)
public func indexOf(elements: Array<T>): Option<Int64>
功能：返回数组中子数组 elements 出现的第一个位置，如果数组中不包含此数组，返回 None。

注意：

当 T 的类型是 Int64 时，此函数的变长参数语法糖版本可能会和 public func indexOf(element: T, fromIndex: Int64): Option<Int64> 产生歧义，根据优先级，当参数数量是 2 个时，会优先调用 public func indexOf(element: T, fromIndex: Int64): Option<Int64>。

参数：

elements: Array<T> - 需要定位的目标数组。
返回值：

Option<Int64> - 数组中子数组 elements 出现的第一个位置，如果数组中不包含此数组，返回 None。
func indexOf(Array<T>, Int64)
public func indexOf(elements: Array<T>, fromIndex: Int64): Option<Int64>
功能：返回数组中在 fromIndex之后，子数组elements 出现的第一个位置，未找到返回 None。

函数会对 fromIndex 范围进行检查，fromIndex 小于 0 时，将会从第 0 位开始搜索，当 fromIndex 大于等于本数组的大小时，结果为 None。

参数：

elements: Array<T> - 需要定位的元素。
fromIndex: Int64 - 开始搜索的起始位置。
返回值：

Option<Int64> - 数组中在 fromIndex之后，子数组 elements 出现的第一个位置，未找到返回 None。
func indexOf(T)
public func indexOf(element: T): Option<Int64>
功能：获取数组中 element 出现的第一个位置，如果数组中不包含此元素，返回 None。

参数：

element: T - 需要定位的元素。
返回值：

Option<Int64> - 数组中 element 出现的第一个位置，如果数组中不包含此元素，返回 None。
func indexOf(T, Int64)
public func indexOf(element: T, fromIndex: Int64): Option<Int64>
功能：返回数组中在 fromIndex之后， element 出现的第一个位置，未找到返回 None。

函数会从下标 fromIndex 开始查找，fromIndex 小于 0 时，将会从第 0 位开始搜索，当 fromIndex 大于等于本数组的大小时，结果为 None。

参数：

element: T - 需要定位的元素。
fromIndex: Int64 - 查找的起始位置。
返回值：

Option<Int64> - 返回数组中在 fromIndex之后， element 出现的第一个位置，未找到返回 None。
func lastIndexOf(Array<T>)
public func lastIndexOf(elements: Array<T>): Option<Int64>
功能：返回数组中子数组 elements 出现的最后一个位置，如果数组中不存在此子数组，返回 None。

参数：

elements: Array<T> - 需要定位的目标数组。
返回值：

Option<Int64> - 数组中 elements 出现的最后一个位置，如果数组中不存在此子数组，返回 None。
func lastIndexOf(Array<T>, Int64)
public func lastIndexOf(elements: Array<T>, fromIndex: Int64): Option<Int64>
功能：从 fromIndex 开始向后搜索，返回数组中子数组 elements 出现的最后一个位置，如果数组中不存在此子数组，返回 None。

函数会对 fromIndex 范围进行检查，fromIndex 小于 0 时，将会从第 0 位开始搜索，当 fromIndex 大于等于本数组的大小时，结果为 None。

参数：

elements: Array<T> - 需要定位的目标数组。
fromIndex: Int64 - 搜索开始的位置。
返回值：

Option<Int64> - 从 fromIndex 开始向后搜索，数组中子数组 elements 出现的最后一个位置，如果数组中不存在此子数组，返回 None。
func lastIndexOf(T)
public func lastIndexOf(element: T): Option<Int64>
功能：返回数组中 element 出现的最后一个位置，如果数组中不存在此元素，返回 None。

参数：

element: T - 需要定位的目标元素。
返回值：

Option<Int64> - 数组中 element 出现的最后一个位置，如果数组中不存在此元素，返回 None
func lastIndexOf(T, Int64)
public func lastIndexOf(element: T, fromIndex: Int64): Option<Int64>
功能：从 fromIndex 开始向后搜索，返回数组中 element 出现的最后一个位置，如果数组中不存在此元素，返回 None。

函数会对 fromIndex 范围进行检查，fromIndex 小于 0 时，将会从第 0 位开始搜索，当 fromIndex 大于等于本数组的大小时，结果为 None。

参数：

element: T - 需要定位的目标元素。
fromIndex: Int64 - 搜索开始的位置。
返回值：

Option<Int64> - 从 fromIndex 开始向后搜索，返回数组中 element 出现的最后一个位置，如果数组中不存在此元素，返回 None。
func trimLeft(Array<T>)
public func trimLeft(prefix: Array<T>): Array<T>
功能：修剪当前数组，去除掉前缀为 prefix 的部分，并且返回当前数组的切片。

参数：

prefix: Array<T> - 要修剪的子串。
返回值：

Array<T> - 修剪后的数组切片。
func trimRight(Array<T>)
public func trimRight(suffix: Array<T>): Array<T>
功能：修剪当前数组，去除掉后缀为 suffix 的部分，并且返回当前数组的切片。

参数：

suffix: Array<T> - 要修剪的子串。
返回值：

Array<T> - 修剪后的数组切片。
operator func !=(Array<T>)
public operator const func !=(that: Array<T>): Bool
功能：判断当前实例与指定 Array<T> 实例是否不等。

参数：

that: Array<T> - 用于与当前实例比较的另一个 Array<T> 实例。
返回值：

Bool - 如果不相等，则返回 true；相等则返回 false。
operator func ==(Array<T>)
public operator const func ==(that: Array<T>): Bool
功能：判断当前实例与指定 Array<T> 实例是否相等。

两个 Array<T> 相等指的是其中的每个元素都相等。

参数：

that: Array<T> - 用于与当前实例比较的另一个 Array<T> 实例。
返回值：

Bool - 如果相等，则返回 true，否则返回 false。
extend<T> Array<T> where T <: ToString
extend<T> Array<T> <: ToString where T <: ToString
功能：为 Array<T> 类型扩展 ToString 接口，支持转字符串操作。

父类型：

ToString
func toString()
public func toString(): String
功能：将数组转换为可输出的字符串。

字符串形如 "[1, 2, 3, 4, 5]"

返回值：

String - 转化后的字符串。