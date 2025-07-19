结构体
struct Array<T>
public struct Array<T> {
    public const init()
    public init(elements: Collection<T>)
    public init(size: Int64, item!: T)
    public init(size: Int64, initElement: (Int64) -> T)
}
功能：仓颉数组类型，用来表示单一类型的元素构成的有序序列。

T 表示数组的元素类型，T 可以是任意类型。

init()
public const init()
功能：构造一个空数组。

init(Collection<T>)
public init(elements: Collection<T>)
功能：根据 Collection 实例创建数组，把 Collection 实例中所有元素存入数组。

参数：

elements: Collection<T> - 根据该 Collection 实例创建数组。
init(Int64, (Int64) -> T)
public init(size: Int64, initElement: (Int64) -> T)
功能：创建指定长度的数组，其中元素根据初始化函数计算获取。

即：将 [0, size) 范围内的值分别传入初始化函数 initElement，执行得到数组对应下标的元素。

参数：

size: Int64 - 数组大小。
initElement: (Int64) ->T - 初始化函数。
异常：

NegativeArraySizeException - 当 size 小于 0，抛出异常。
init(Int64, T)
public init(size: Int64, item!: T)
功能：构造一个指定长度的数组，其中元素都用指定初始值进行初始化。

注意：

该构造函数不会拷贝 item， 如果 item 是一个引用类型，构造后数组的每一个元素都将指向相同的引用。

参数：

size: Int64 - 数组大小，取值范围为 [0, Int64.Max]。
item!: T - 数组元素初始值。
异常：

NegativeArraySizeException - 当 size 小于 0，抛出异常。
func clone()
public func clone(): Array<T>
功能：克隆数组，将对数组数据进行深拷贝。

返回值：

Array<T> - 克隆得到的新数组。
func clone(Range<Int64>)
public func clone(range: Range<Int64>) : Array<T>
功能：克隆数组的指定区间。

注意：

如果参数 range 是使用 Range 构造函数构造的 Range 实例，有如下行为：
start 的值就是构造函数传入的值本身，不受构造时传入的 hasStart 的值的影响。
hasEnd 为 false 时，end 值不生效，且不受构造时传入的 isClosed 的值的影响，数组切片取到原数组最后一个元素。
range 的步长只能为 1。
参数：

range: Range<Int64> - 克隆的区间。
返回值：

Array<T> - 克隆得到的新数组。
异常：

IndexOutOfBoundsException - range 超出数组范围时，抛出异常。
示例：

main() {
    let arr = [0, 1, 2, 3, 4, 5]
    let new = arr.clone(1..4)
    println(new)
}
运行结果：

[1, 2, 3]
func concat(Array<T>)
public func concat(other: Array<T>): Array<T>
功能：该函数将创建一个新的数组，数组内容是当前数组后面串联 other 指向的数组。

参数：

other: Array<T> - 串联到当前数组末尾的数组。
返回值：

Array<T> - 串联得到的新数组。
示例：

main() {
    let arr = [0, 1, 2, 3, 4, 5]
    let new = arr.concat([6, 7, 8, 9, 10])
    println(new)
}
运行结果：

[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
func copyTo(Array<T>, Int64, Int64, Int64)
public func copyTo(dst: Array<T>, srcStart: Int64, dstStart: Int64, copyLen: Int64): Unit
功能：将当前数组中的一段数据拷贝到目标数组中。

参数：

dst: Array<T> - 目标数组。
srcStart: Int64 - 从 this 数组的 srcStart 下标开始拷贝，取值范围为 [0, this.size)。
dstStart: Int64 - 从目标数组的 dstStart 下标开始写入，取值范围为 [0, dst.size)。
copyLen: Int64 - 拷贝数组的长度，取值要求为 copyLen + srcStart < this.size，copyLen + dstStart < dst.size。
异常：

IllegalArgumentException - copyLen 小于 0 则抛出此异常。
IndexOutOfBoundsException - 如果参数不满足上述取值范围，抛出此异常。
示例：

main() {
    let arr = [0, 1, 2, 3, 4, 5]
    let new = [0, 0, 0, 0, 0, 0]
    arr.copyTo(new, 2, 2, 4)
    println(new)
}
运行结果：

[0, 0, 2, 3, 4, 5]
func get(Int64)
public func get(index: Int64): Option<T>
功能：获取数组中下标 index 对应的元素。

该函数结果将用 Option 封装，如果 index 越界，将返回 None。

也可以通过 [] 操作符获取数组指定下标的元素，该接口将在 index 越界时抛出异常。

参数：

index: Int64 - 要获取的值的下标。
返回值：

Option<T> - 当前数组中下标 index 对应的值。
func reverse()
public func reverse(): Unit
功能：反转数组，将数组中元素的顺序进行反转。

示例：

main() {
    let arr = [0, 1, 2, 3, 4, 5]
    arr.reverse()
    println(arr)
}
运行结果：

[5, 4, 3, 2, 1, 0]
func set(Int64, T)
public func set(index: Int64, element: T): Unit
功能：修改数组中下标 index 对应的值。

也可以通过 [] 操作符完成对指定下标元素的修改，这两个函数的行为一致。

参数：

index: Int64 - 需要修改的值的下标，取值范围为 [0..this.size]。
element: T - 修改的目标值。
异常：

IndexOutOfBoundsException - 如果 index 小于 0 或者大于或等于 Array 的长度，抛出异常。
func slice(Int64, Int64)
public func slice(start: Int64, len: Int64): Array<T>
功能：获取数组切片。

注意：

切片不会对数组数据进行拷贝，是对原数据特定区间的引用。

参数：

start: Int64 - 切片的起始位置，取值需大于 0，且 start + len 小于等于当前 Array 实例的长度。
len: Int64 - 切片的长度，取值需大于 0。
返回值：

Array<T> - 返回切片后的数组。
异常：

IndexOutOfBoundsException - 如果参数不符合上述取值范围，抛出异常。
operator func [](Int64)
public operator func [](index: Int64): T
功能：获取数组下标 index 对应的值。

该函数中如果 index 越界，将抛出异常。

也可以通过 get 函数获取数组指定下标的元素，get 函数将在 index 越界时返回 None。

参数：

index: Int64 - 要获取的值的下标，取值范围为 [0, Int64.Max]。
返回值：

T - 数组中下标 index 对应的值。
异常：

IndexOutOfBoundsException - 如果 index 小于 0，或大于等于数组长度，抛出异常。
operator func [](Int64, T)
public operator func [](index: Int64, value!: T): Unit
功能：修改数组中下标 index 对应的值。

参数：

index: Int64 - 需要修改值的下标，取值范围为 [0, Int64.Max]。
value!: T - 修改的目标值。
异常：

IndexOutOfBoundsException - 如果 index 小于 0，或大于等于数组长度，抛出异常。
operator func [](Range<Int64>)
public operator func [](range: Range<Int64>): Array<T>
功能：根据给定区间获取数组切片。

注意：

如果参数 range 是使用 Range 构造函数构造的 Range 实例，有如下行为：
start 的值就是构造函数传入的值本身，不受构造时传入的 hasStart 的值的影响。
hasEnd 为 false 时，end 值不生效，且不受构造时传入的 isClosed 的值的影响，该数组切片取到原数组最后一个元素。
range 的步长只能为 1。
参数：

range: Range<Int64> - 切片的范围，range 表示的范围不能超过数组范围。
返回值：

Array<T> - 数组切片。
异常：

IllegalArgumentException - 如果 range 的步长不等于 1，抛出异常。
IndexOutOfBoundsException - 如果 range 表示的数组范围无效，抛出异常。
示例：

main() {
    let arr = [0, 1, 2, 3, 4, 5]
    let slice = arr[1..4]
    arr[3] = 10
    println(slice)
}
运行结果：

[1, 2, 10]
operator func [](Range<Int64>, Array<T>)
public operator func [](range: Range<Int64>, value!: Array<T>): Unit
功能：用指定的数组对本数组一个连续范围的元素赋值。

range 表示的区见的长度和目标数组 value 的大小需相等。

注意：

如果参数 range 是使用 Range 构造函数构造的 Range 实例，有如下行为：
start 的值就是构造函数传入的值本身，不受构造时传入的 hasStart 的值的影响。
hasEnd 为 false 时，end 值不生效，且不受构造时传入的 isClosed 的值的影响，该数组切片取到原数组最后一个元素。
range 的步长只能为 1。
参数：

range: Range<Int64> - 需要修改的数组范围，range 表示的范围不能超过数组范围。
value!: Array<T> - 修改的目标值。
异常：

IllegalArgumentException - 如果 range 的步长不等于 1，或 range 长度不等于 value 长度，抛出异常。
IndexOutOfBoundsException - 如果 range 表示的数组范围无效，抛出异常。
示例：

main() {
    let arr = [0, 1, 2, 3, 4, 5]
    arr[1..3] = [10, 11]
    println(arr)
}
运行结果：

[0, 10, 11, 3, 4, 5]