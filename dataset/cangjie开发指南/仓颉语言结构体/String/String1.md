func lastIndexOf(Byte)
public func lastIndexOf(b: Byte): Option<Int64>
功能：返回指定字节 b 最后一次出现的在原字符串内的索引。

参数：

b: Byte - 待搜索的字节。
返回值：

Option<Int64> - 如果原字符串中包含此字节，返回其最后一次出现的索引，否则返回 None。
func lastIndexOf(Byte, Int64)
public func lastIndexOf(b: Byte, fromIndex: Int64): Option<Int64>
功能：从原字符串 fromIndex 索引开始搜索，返回指定 UTF-8 编码字节 b 最后一次出现的在原字符串内的索引。

参数：

b: Byte - 待搜索的字节。
fromIndex: Int64 - 以指定的索引 fromIndex 开始搜索。
返回值：

Option<Int64> - 如果搜索成功，返回指定字节最后一次出现的索引，否则返回 None。特别地，当 fromIndex 小于零，效果同 0，当 fromIndex 大于等于原字符串长度，返回 None。
func lastIndexOf(String)
public func lastIndexOf(str: String): Option<Int64>
功能：返回指定字符串 str 最后一次出现的在原字符串的起始索引。

参数：

str: String - 待搜索的字符串。
返回值：

Option<Int64> - 如果原字符串中包含 str 字符串，返回其最后一次出现的索引，否则返回 None。
func lastIndexOf(String, Int64)
public func lastIndexOf(str: String, fromIndex: Int64): Option<Int64>
功能：从原字符串指定索引开始搜索，获取指定字符串 str 最后一次出现的在原字符串的起始索引。

参数：

str: String - 待搜索的字符串。
fromIndex: Int64 - 以指定的索引 fromIndex 开始搜索。
返回值：

Option<Int64> - 如果这个字符串在位置 fromIndex 及其之后没有出现，则返回 None。特别地，当 str 是空字符串时，如果 fromIndex 大于 0，返回 None，否则返回 Some(0)，当 fromIndex 小于零，效果同 0，当 fromIndex 大于等于原字符串长度返回 None。
func lazySplit(String, Bool)
public func lazySplit(str: String, removeEmpty!: Bool = false): Iterator<String>
功能：对原字符串按照字符串 str 分隔符分割，该函数不立即对字符串进行分割，而是返回迭代器，使用迭代器进行遍历时再实际执行分隔操作。

当 str 未出现在原字符串中，返回大小为 1 的字符串迭代器，唯一的元素为原字符串。

参数：

str: String - 字符串分隔符。
removeEmpty!: Bool - 移除分割结果中的空字符串，默认值为 false。
返回值：

Iterator<String> - 分割后的字符串迭代器。
func lazySplit(String, Int64, Bool)
public func lazySplit(str: String, maxSplits: Int64, removeEmpty!: Bool = false): Iterator<String>
功能：对原字符串按照字符串 str 分隔符分割，该函数不立即对字符串进行分割，而是返回迭代器，使用迭代器进行遍历时再实际执行分隔操作。

当 maxSplit 为 0 时，返回空的字符串迭代器；
当 maxSplit 为 1 时，返回大小为 1 的字符串迭代器，唯一的元素为原字符串；
当 maxSplit 为负数时，直接返回分割后的字符串迭代器；
当 maxSplit 大于完整分割出来的子字符串数量时，返回完整分割的字符串迭代器；
当 str 未出现在原字符串中，返回大小为 1 的字符串迭代器，唯一的元素为原字符串；
当 str 为空时，对每个字符进行分割；当原字符串和分隔符都为空时，返回空字符串迭代器。
参数：

str: String - 字符串分隔符。
maxSplits: Int64 - 最多分割为 maxSplit 个子字符串。
removeEmpty!: Bool - 移除分割结果中的空字符串，默认值为 false。
返回值：

Iterator<String> - 分割后的字符串迭代器。
func lines()
public func lines(): Iterator<String>
功能：获取字符串的行迭代器，每行都由换行符进行分隔，换行符是 \n \r \r\n 之一，结果中每行不包括换行符。

返回值：

Iterator<String> - 字符串的行迭代器。
func padLeft(Int64, String)
public func padLeft(totalWidth: Int64, padding!: String = " "): String
功能：按指定长度右对齐原字符串，如果原字符串长度小于指定长度，在其左侧添加指定字符串。

当指定长度小于字符串长度时，返回字符串本身，不会发生截断；当指定长度大于字符串长度时，在左侧添加 padding 字符串，当 padding 长度大于 1 时，返回字符串的长度可能大于指定长度。

参数：

totalWidth: Int64 - 指定对齐后字符串长度，取值需大于等于 0。
padding!: String - 当长度不够时，在左侧用指定的字符串 padding 进行填充
返回值：

String - 填充后的字符串。
异常：

IllegalArgumentException - 如果 totalWidth 小于 0，抛出异常。
func padRight(Int64, String)
public func padRight(totalWidth: Int64, padding!: String = " "): String
功能：按指定长度左对齐原字符串，如果原字符串长度小于指定长度，在其右侧添加指定字符串。

当指定长度小于字符串长度时，返回字符串本身，不会发生截断；当指定长度大于字符串长度时，在右侧添加 padding 字符串，当 padding 长度大于 1 时，返回字符串的长度可能大于指定长度。

参数：

totalWidth: Int64 - 指定对齐后字符串长度，取值需大于等于 0。
padding!: String - 当长度不够时，在右侧用指定的字符串 padding 进行填充。
返回值：

String - 填充后的字符串。
异常：

IllegalArgumentException - 如果 totalWidth 小于 0，抛出异常。
func rawData()
public unsafe func rawData(): Array<Byte>
功能：获取字符串的 UTF-8 编码的原始字节数组。

注意：

用户不应该对获取的数组进行修改，这将破坏字符串的不可变性。

返回值：

Array<Byte> - 当前字符串对应的原始字节数组。
func releaseRaw(CPointerHandle<UInt8>)
public unsafe func releaseRaw(cp: CPointerHandle<UInt8>): Unit
功能：释放 getRaw 函数获取的指针。

注意：

释放时只能释放同一个 String 获取的指针，如果释放了其他 String 获取的指针，会出现不可预知的错误。

参数：

cp: CPointerHandle<UInt8> - 待释放的指针实例。
func replace(String, String)
public func replace(old: String, new: String): String
功能：使用新字符串替换原字符串中旧字符串。

参数：

old: String - 旧字符串。
new: String - 新字符串。
返回值：

String - 替换后的新字符串。
异常：

OutOfMemoryError - 如果此函数分配内存时产生错误，抛出异常。
func runes()
public func runes(): Iterator<Rune>
功能：获取字符串的 Rune 迭代器。

返回值：

Iterator<Rune> - 字符串的 Rune 迭代器。
异常：

IllegalArgumentException - 使用 for-in 或者 next() 方法遍历迭代器时，如果读取到非法字符，抛出异常。
func split(String, Bool)
public func split(str: String, removeEmpty!: Bool = false): Array<String>
功能：对原字符串按照字符串 str 分隔符分割，指定是否删除空串。

当 str 未出现在原字符串中，返回长度为 1 的字符串数组，唯一的元素为原字符串。

参数：

str: String - 字符串分隔符。
removeEmpty!: Bool - 移除分割结果中的空字符串，默认值为 false。
返回值：

Array<String> - 分割后的字符串数组。
func split(String, Int64, Bool)
public func split(str: String, maxSplits: Int64, removeEmpty!: Bool = false): Array<String>
功能：对原字符串按照字符串 str 分隔符分割，指定最多分隔子串数，以及是否删除空串。

当 maxSplit 为 0 时，返回空的字符串数组；
当 maxSplit 为 1 时，返回长度为 1 的字符串数组，唯一的元素为原字符串；
当 maxSplit 为负数时，返回完整分割后的字符串数组；
当 maxSplit 大于完整分割出来的子字符串数量时，返回完整分割的字符串数组；
当 str 未出现在原字符串中，返回长度为 1 的字符串数组，唯一的元素为原字符串；
当 str 为空时，对每个字符进行分割；当原字符串和分隔符都为空时，返回空字符串数组。
参数：

str: String - 字符串分隔符。
maxSplits: Int64 - 最多分割为 maxSplit 个子字符串。
removeEmpty!: Bool - 移除分割结果中的空字符串，默认值为 false。
返回值：

Array<String> - 分割后的字符串数组。
func startsWith(String)
public func startsWith(prefix: String): Bool
功能：判断原字符串是否以 prefix 字符串为前缀。

参数：

prefix: String - 被判断的前缀字符串。
返回值：

Bool - 如果字符串 str 是原字符串的前缀，返回 true，否则返回 false，特别地，如果 str 字符串长度为 0，返回 true。
func toArray()
public func toArray(): Array<Byte>
功能：获取字符串的 UTF-8 编码的字节数组。

返回值：

Array<Byte> - 字符串的 UTF-8 编码的字节数组。