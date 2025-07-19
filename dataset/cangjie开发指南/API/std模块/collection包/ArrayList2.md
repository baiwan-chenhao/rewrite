func remove(Range<Int64>)
public func remove(range: Range<Int64>): Unit
功能：删除此 ArrayList 中 Range 范围所包含的所有元素。

注意：

如果参数 range 是使用 Range 构造函数构造的 Range 实例，hasEnd 为 false 时，end 值不生效，且不受构造时传入的 isClosed 的值的影响，数组切片取到原数组最后一个元素。

参数：

range: Range<Int64> - 需要被删除的元素的范围。
异常：

IllegalArgumentException - 当 range 的 step 不等于 1 时抛出异常。
IndexOutOfBoundsException - 当 range 的 start 或 end 小于 0，或 end 大于 Array 的长度时抛出。
func removeIf((T) -> Bool)
public func removeIf(predicate: (T) -> Bool): Unit
功能：删除此 ArrayList 中满足给定 lambda 表达式或函数的所有元素。

参数：

predicate: (T) ->Bool - 传递判断删除的条件。
func reserve(Int64)
public func reserve(additional: Int64): Unit
功能：增加此 ArrayList 实例的容量。

将 ArrayList 扩容 additional 大小，当 additional 小于等于零时，不发生扩容；当 ArrayList 剩余容量大于等于 additional 时，不发生扩容；当 ArrayList 剩余容量小于 additional 时，取（原始容量的1.5倍向下取整）与（additional + 已使用容量）两个值中的最大值进行扩容。

参数：

additional: Int64 - 将要扩容的大小。
异常：

OverflowException - 当additional + 已使用容量超过Int64.Max时，抛出异常。
func reverse()
public func reverse(): Unit
功能：反转此 ArrayList 中元素的顺序。

func set(Int64, T)
public func set(index: Int64, element: T): Unit
功能：将此 ArrayList 中指定位置的元素替换为指定的元素。

参数：

index: Int64 - 要设置的索引值。
element: T - T 类型元素。
异常：

IndexOutOfBoundsException - 当 index 小于 0 或者大于等于 ArrayList 中的元素数量时，抛出异常。
示例：

使用示例见 ArrayList 的 get/set 函数。

func slice(Range<Int64>)
public func slice(range: Range<Int64>): ArrayList<T>
功能：以传入参数 range 作为索引，返回索引对应的 ArrayList<T>。

注意：

如果参数 range 是使用 Range 构造函数构造的 Range 实例，有如下行为：

start 的值就是构造函数传入的值本身，不受构造时传入的 hasStart 的值的影响。
hasEnd 为 false 时，end 值不生效，且不受构造时传入的 isClosed 的值的影响，该数组切片取到原数组最后一个元素。
参数：

range: Range<Int64> - 传递切片的范围。
返回值：

ArrayList<T> - 切片所得的数组。
异常：

IllegalArgumentException - 当 range.step 不等于 1 时，抛出异常。
IndexOutOfBoundsException - 当 range 无效时，抛出异常。
示例：

使用示例见 ArrayList 的 remove/clear/slice 函数。

func sortBy(Bool, (T, T) -> Ordering)
public func sortBy(stable!: Bool = false, comparator!: (T, T) -> Ordering): Unit
功能：对数组中的元素进行排序。

通过传入的比较函数，根据其返回值 Ordering 类型的结果，可对数组进行自定义排序comparator: (t1: T, t2: T) -> Ordering，如果 comparator 的返回值为 Ordering.GT，排序后 t1 在 t2后；如果 comparator 的返回值为 Ordering.LT，排序后 t1 在t2 前；如果 comparator 的返回值为 Ordering.EQ，且为稳定排序，那么 t1 在 t2 之前； 如果 comparator 的返回值为 Ordering.EQ，且为不稳定排序，那么 t1，t2 顺序不确定。

参数：

stable!: Bool - 是否使用稳定排序。
comparator!: (T, T) ->Ordering - (T, T) -> Ordering 类型。
func toArray()
public func toArray(): Array<T>
功能：返回一个数组，其中包含此列表中按正确顺序排列的所有元素。

返回值：

Array<T> - T 类型数组。
operator func [](Int64)
public operator func [](index: Int64): T
功能：操作符重载 - get。

参数：

index: Int64 - 表示 get 接口的索引。
返回值：

T - 索引位置的元素的值。
异常：

IndexOutOfBoundsException - 当 index 超出范围时，抛出异常。
operator func [](Int64, T)
public operator func [](index: Int64, value!: T): Unit
功能：操作符重载 - set，通过下标运算符用指定的元素替换此列表中指定位置的元素。

参数：

index: Int64 - 要设置的索引值。
value!: T - 要设置的 T 类型的值。
异常：

IndexOutOfBoundsException - 当 index 超出范围时，抛出异常。
operator func [](Range<Int64>)
public operator func [](range: Range<Int64>): ArrayList<T>
功能：运算符重载 - 切片。

注意：

如果参数 range 是使用 Range 构造函数构造的 Range 实例，有如下行为：

start 的值就是构造函数传入的值本身，不受构造时传入的 hasStart 的值的影响。
hasEnd 为 false 时，end 值不生效，且不受构造时传入的 isClosed 的值的影响，数组切片取到原数组最后一个元素。
切片操作返回的 ArrayList 为全新的对象，与原 ArrayList 无引用关系。

参数：

range: Range<Int64> - 传递切片的范围。
返回值：

ArrayList<T> - 切片所得的数组。
异常：

IllegalArgumentException - 当 range.step 不等于 1 时，抛出异常。
IndexOutOfBoundsException - 当 range 无效时，抛出异常。
extend<T> ArrayList<T> <: Equatable<ArrayList<T>> where T <: Equatable<T>
extend<T> ArrayList<T> <: Equatable<ArrayList<T>> where T <: Equatable<T>
功能：为 ArrayList<T> 类型扩展 Equatable<ArrayList<T>> 接口，支持判等操作。

父类型：

Equatable<ArrayList<T>>
operator func ==(ArrayList<T>)
public operator func ==(that: ArrayList<T>): Bool
功能：判断当前实例与参数指向的 ArrayList 实例是否相等。

两个数组相等指的是两者对应位置的元素分别相等。

参数：

that: ArrayList<T> - 被比较的对象。
返回值：

Bool - 如果相等，则返回 true，否则返回 false。
operator func !=(ArrayList<T>)
public operator func !=(that: ArrayList<T>): Bool
功能：判断当前实例与参数指向的 ArrayList 实例是否不等。

参数：

that: ArrayList<T> - 被比较的对象。
返回值：

Bool - 如果不等，则返回 true，否则返回 false。
func contains(T)
public func contains(element: T): Bool
功能：判断当前数组中是否含有指定元素 element。

参数：

element: T - 待寻找的元素。
返回值：

Bool - 如果数组中包含指定元素，返回 true，否则返回 false。
extend<T> ArrayList<T> <: SortExtension where T <: Comparable<T>
extend<T> ArrayList<T> <: SortExtension where T <: Comparable<T>
功能：为 ArrayList<T> 扩展 SortExtension 接口，支持数组排序。

父类型：

SortExtension
func sort(Bool)
public func sort(stable!: Bool = false): Unit
功能：将当前数组内元素以升序的方式排序。

参数：

stable!: Bool - 是否使用稳定排序。
func sortDescending(Bool)
public func sortDescending(stable!: Bool = false): Unit
功能：将当前数组内元素以降序的方式排序。

参数：

stable!: Bool - 是否使用稳定排序。
extend<T> ArrayList<T> <: ToString where T <: ToString
extend<T> ArrayList<T> <: ToString where T <: ToString
功能：为 ArrayList<T> 扩展 ToString 接口，支持转字符串操作。

父类型：

ToString
func toString()
public func toString(): String
功能：将当前数组转换为字符串。

该字符串包含数组内每个元素的字符串表示，形如："[elem1, elem2, elem3]"。

返回值：

String - 转换得到的字符串。