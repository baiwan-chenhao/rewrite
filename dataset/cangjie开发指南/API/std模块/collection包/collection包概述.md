std.collection 包
功能介绍
collection 包提供了常见数据结构的高效实现、相关抽象的接口的定义以及在集合类型中常用的函数功能。

本包实现了以下常用的数据结构：

ArrayList：变长的连续数组，在需要存储不确定数量的数据，或者需要根据运行时的条件动态调整数组大小时使用 ArrayList，使用 ArrayList 可能会导致内存分配和释放的开销增加，因此需要谨慎使用。

LinkedList：链表结构， LinkedList 的优点是它可以动态地添加或删除元素，而不需要移动其他元素。这使得它在需要频繁添加或删除元素的情况下非常有用。它还可以轻松地进行修改或删除操作，并且可以在列表中存储多个元素。 LinkedList 的缺点是它需要额外的内存来存储每个元素的引用，这可能会导致内存浪费。

HashMap：哈希表，它存储键值对，并且可以根据键快速访问值。在需要使用映射关系并且需要快速查找时使用。

HashSet：基于哈希表实现的集合数据结构，它可以用于快速检索和删除元素，具有高效的插入、删除和查找操作。

TreeMap：基于红黑树实现的有序映射表。通常情况下，当需要将元素按照自然顺序或者自定义顺序进行排序时，可以使用TreeMap。

collection 包提供的集合类型都不支持并发安全，并发安全的集合请见 collection.concurrent 包。

API 列表
函数
函数名	功能
all<T>((T) -> Bool)	判断迭代器所有元素是否都满足条件。
any<T>((T) -> Bool)	判断迭代器是否存在任意一个满足条件的元素。
at<T>(Int64)	获取迭代器指定位置的元素。
collectArrayList<T>(Iterable<T>)	将一个迭代器转换成 ArrayList 类型。
collectArray<T>(Iterable<T>)	将一个迭代器转换成 Array 类型。
collectHashMap<K, V>(Iterable<(K, V)>) where K <: Hashable & Equatable<K>	将一个迭代器转换成 HashMap 类型。
collectHashSet<T>(Iterable<T>) where T <: Hashable & Equatable<T>	将一个迭代器转换成 HashSet 类型。
collectString<T>(String) where T <: ToString	将一个对应元素实现了 ToString 接口的迭代器转换成 String 类型。
concat<T>(Iterable<T>)	串联两个迭代器。
contains<T>(T) where T <: Equatable<T>	遍历所有元素，判断是否包含指定元素并返回该元素。
count<T>(Iterable<T>)	统计迭代器包含元素数量。
enumerate<T>(Iterable<T>)	用于获取带索引的迭代器。
filter<T>((T) -> Bool)	筛选出满足条件的元素。
filterMap<T, R>((T) -> ?R)	同时进行筛选操作和映射操作，返回一个新的迭代器。
first<T>(Iterable<T>)	获取头部元素。
flatMap<T, R>( (T) -> Iterable<R>)	创建一个带 flatten 功能的映射。
flatten<T, R>(Iterable<T>) where T <: Iterable<R>	将嵌套的迭代器展开一层。
fold<T, R>(R, (R, T) -> R)	使用指定初始值，从左向右计算。
forEach<T>((T) -> Unit)	遍历所有元素，指定给定的操作。
inspect<T>((T) -> Unit)	迭代器每次调用 next() 对当前元素执行额外操作（不会消耗迭代器中元素）。
isEmpty<T>(Iterable<T>)	判断迭代器是否为空。
last<T>(Iterable<T>)	获取尾部元素。
map<T, R>((T) -> R)	创建一个映射。
max<T>(Iterable<T>) where T <: Comparable<T>	筛选最大的元素。
min<T>(Iterable<T>) where T <: Comparable<T>	筛选最小的元素。
none<T>((T) -> Bool)	判断迭代器是否都不满足条件。
reduce<T>((T, T) -> T)	使用第一个元素作为初始值，从左向右计算。
skip<T>(Int64)	从迭代器跳过特定个数。
step<T>(Int64)	迭代器每次调用 next() 跳过特定个数。
take<T>(Int64)	从迭代器取出特定个数。
zip<T, R>(Iterable<R>)	将两个迭代器合并成一个（长度取决于短的那个迭代器）。
接口
接口名	功能
Map<K, V> where K <: Equatable<K>	提供了一种将键映射到值的方式。
Set<T> where T <: Equatable<T>	不包含重复元素的集合。
EquatableCollection<T> where T <: Equatable<T>	定义了可以进行比较的集合类型。
类
类名	功能
ArrayList<T>	提供可变长度的数组的功能。
ArrayListIterator<T>	此类主要实现 ArrayList<T> 的迭代器功能。
HashMapIterator<K, V> where K <: Hashable & Equatable<K>	此类主要实现 HashMap 的迭代器功能。
HashMap<K, V> where K <: Hashable & Equatable<K>	Map<K, V> where K <: Equatable<K> 接口的哈希表实现。
HashSet<T> where T <: Hashable & Equatable<T>	基于 HashMap<K, V> where K <: Hashable & Equatable<K> 实现的 Set<T> where T <: Equatable<T> 接口的实例。
LinkedListNode<T>	LinkedList<T> 上的节点。
LinkedList<T>	实现双向链表的数据结构。
TreeMap<K, V> where K <: Comparable<K>	基于平衡二叉搜索树实现的 Map<K, V> where K <: Equatable<K> 接口实例。
结构体
结构体名	功能
EntryView<K, V> where K <: Hashable & Equatable<K>	HashMap<K, V> where K <: Hashable & Equatable<K> 中某一个键的视图。
TreeMapNode<K, V> where K <: Comparable<K>	TreeMap<K, V> where K <: Comparable<K> 的节点结构。
异常类
异常类名	功能
ConcurrentModificationException	并发修改异常类。