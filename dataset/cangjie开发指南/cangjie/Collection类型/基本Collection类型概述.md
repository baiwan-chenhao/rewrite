基础 Collection 类型概述
本章我们来看看仓颉中常用的几种基础 Collection 类型，包含 Array、ArrayList、HashSet、HashMap。

我们可以在不同的场景中选择适合我们业务的类型：

Array：如果我们不需要增加和删除元素，但需要修改元素，就应该使用它。
ArrayList：如果我们需要频繁对元素增删查改，就应该使用它。
HashSet：如果我们希望每个元素都是唯一的，就应该使用它。
HashMap：如果我们希望存储一系列的映射关系，就应该使用它。
下表是这些类型的基础特性：

类型名称	元素可变	增删元素	元素唯一性	有序序列
Array<T>	Y	N	N	Y
ArrayList<T>	Y	Y	N	Y
HashSet<T>	N	Y	Y	N
HashMap<K, V>	K: N, V: Y	Y	K: Y, V: N	N