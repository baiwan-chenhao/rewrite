import shutil
import subprocess
import os
from datetime import datetime


class CodeEvaluator:
    # todo: 多线程支持
    def __init__(self, timeout=4):
        """
        :param timeout: 超时，以秒为单位。
        failed : 是否失败
        result : 运行结果
        """
        self.timeout = timeout
        self.failed = None
        self.result = None
        self.is_timeouted = False
        self.encoding = "utf-8"

    def get_temp_filename(self):
        filename = str(os.getpid()) + "-" + datetime.now().strftime("%M-%S")
        return filename

    def run(self, code_or_path=".code_evaluating.cj"):
        res = None
        self.is_timeouted = False
        if os.path.isfile(code_or_path):
            with open(code_or_path, "r", encoding='utf-8') as f:
                code = f.read()
        else:
            code = code_or_path
        try:
            res = self.run_code(code)
        except subprocess.TimeoutExpired:
            self.failed = True
            self.result = "timeout"
            self.is_timeouted = True
        return res

    def run_code(self, code):
        raise NotImplemented

    def is_valid_code(self, code):
        try:
            self.run_code(code)
        except subprocess.TimeoutExpired:
            return True
        return not self.failed

    def run_with_timeout(self, command):
        class Return:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        # 启动子进程
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True,
                                   encoding=self.encoding)

        # 等待子进程完成，超时后抛出 TimeoutExpired 异常
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            process.kill()
            raise e
        return_code = process.returncode
        return Return(return_code, stdout, stderr)


"""
+ 编译成功，返回运行结果
+ 编译失败，返回编译结果
"""

def remove(file):
    if os.path.isfile(file):
        try:
            os.remove(file)
        except PermissionError:
            pass


class CJCodeEvaluator(CodeEvaluator):
    loaded_lib = ["import std.collection.*\n", "import std.math.*\n"]

    def run_code(self, code):
        """
        :param code:
        :return:
        """
        filename = self.get_temp_filename()
        if "main()" not in code:
            code += "\n\nmain(){\n}"
        for lib in self.loaded_lib:
            if lib not in code:
                code = lib + code
        with open(f"{filename}.cj", "w", encoding='utf-8') as f:
            f.write(code)
        args = f"cjc {filename}.cj -o {filename}.exe"  #  --output-type={output_type}"
        completed = subprocess.run(args=args, capture_output=True, shell=False, timeout=self.timeout, encoding="utf-8")
        # completed = self.run_with_timeout(args)   that code has trouble in CJ code running.
        if completed.returncode == 0:
            completed1 = subprocess.run(f"{filename}.exe", capture_output=True, shell=False, timeout=self.timeout, encoding="utf-8")
            # completed1 = self.run_with_timeout("code_evaluating.exe")
            if completed1.returncode == 0:
                self.result = completed1.stdout #.decode("utf-8")
                self.failed = False
            else:
                self.result = completed1.stderr # .decode("utf-8")
                self.failed = True
        else:
            self.failed = True
            self.result = completed.stderr # .decode("utf-8")
        remove(f"{filename}.exe")
        remove(f"{filename}.cj")

    def test(self):
        self.run_code("main(){\nprintln('hello world')\n}")
        print("main 正常", self)
        self.run_code("main(){\nprintln(1/0)\n}")
        print("main 运行时", self)
        self.run_code("main(){\nwf\n}")
        print("main 编译", self)

        self.run_code("func man(){\nprintln('hello world')\n}")
        print("func 正常", self)
        self.run_code("func man(){\nprintln(1/0)\n}")
        print("func 运行时", self)
        self.run_code("func man(){\nwf\n}")
        print("func 编译", self)

    def __str__(self):
        return f"result:{self.result}\nfailed:{self.failed}"


class PYCodeEvaluator(CodeEvaluator):
    loaded_lib = ["from typing import *", "from collections import *"]

    def run_code(self, code):
        """
        :param code:
        :return:
        """
        for lib in self.loaded_lib:
            if lib not in code:
                code = lib + "\n" + code
        with open(".code_evaluating.py", "w", encoding='utf-8') as f:
            f.write(code)
        args = f"python .code_evaluating.py"
        completed = subprocess.run(args=args, capture_output=True, shell=False, timeout=self.timeout)
        if completed.returncode == 0:
            self.result = completed.stdout.decode("utf-8")
            self.failed = False
        else:
            self.result = completed.stderr.decode("utf-8")
            self.failed = True

    def test(self):
        self.run_code("print(0)")
        print("正常", self)
        self.run_code("print(")
        print("语法", self)
        self.run_code("print(1/0)")
        print("运行时", self)

    def __str__(self):
        return f"result:{self.result}\nfailed:{self.failed}"

cj_coder = CJCodeEvaluator()


class ScalaCodeEvaluator(CodeEvaluator):
    def run_code(self, code, warpped=True):
        """
        :param code:
        :param warpped: 是否包裹代码。默认输入的`code`是
        :return:
        """
        with open("CodeEvaluating.scala", "w", encoding='utf-8') as f:
            f.write(code)
        import os
        # os.system("scalac CodeEvaluating.scala")
        # os.system("scala HelloWorld")
        args = [r"C:\Program Files (x86)\scala\bin\scalac.bat", "CodeEvaluating.scala"]
        completed = self.run_with_timeout(args)
        # subprocess.run(args=args, capture_output=True, shell=False, timeout=self.timeout)
        if completed.returncode == 0:
            completed1 = self.run_with_timeout([r"C:\Program Files (x86)\scala\bin\scala.bat",
                                                "CodeEvaluating.scala"])
            if completed1.returncode == 0:
                self.result = completed1.stdout  # .decode("utf-8")
                self.failed = False
            else:
                self.result = completed1.stderr  # .decode("utf-8")
                self.failed = True
        else:
            self.failed = True
            self.result = completed.stderr  # .decode("utf-8")

    def test(self):
        self.run_code("print(0)")
        print("正常", self)
        self.run_code("print(")
        print("语法", self)
        self.run_code("print(1/0)")
        print("运行时", self)

    def __str__(self):
        return f"result:{self.result}\nfailed:{self.failed}"


class ErlangCodeEvaluator(CodeEvaluator):
    def run_code(self, code):
        with open("CodeEvaluating.erl", "w", encoding='utf-8') as f:
            f.write(code)

        args = [r"C:\Program Files\Erlang OTP\bin\escript", "CodeEvaluating.erl"]
        completed = self.run_with_timeout(args)
        if completed.returncode == 0:
            self.result = completed.stdout  # .decode()
            self.failed = False
        else:
            self.failed = True
            self.result = completed.stderr  # .decode()

        os.remove("CodeEvaluating.erl")

    def __str__(self):
        return f"result:{self.result}\nfailed:{self.failed}"


class JavaCodeEvaluator(CodeEvaluator):
    exe_path = r"D:\app\java\jdk-16.0.1\bin\javac.exe", r"D:\app\java\jdk-16.0.1\bin\java.exe"

    def __init__(self, timeout=4):
        super().__init__(timeout=timeout)
        self.encoding = "gbk"

    def run_code(self, code, warpped=True):
        """
        :param code: 要运行的Java代码（字符串形式）
        :param warpped: 是否需要包裹代码（默认是已经完整的）
        :return:
        """
        # 写入Java源文件
        with open("CodeEvaluating.java", "w", encoding='utf-8') as f:
            f.write(code)

        # 编译 Java 文件
        args_compile = [self.exe_path[0], "CodeEvaluating.java"]
        completed_compile = self.run_with_timeout(args_compile)

        if completed_compile.returncode == 0:
            # 如果编译成功，运行 Java 类
            args_run = [self.exe_path[1], "CodeEvaluating"]
            completed_run = self.run_with_timeout(args_run)
            if completed_run.returncode == 0:
                self.result = completed_run.stdout
                self.failed = False
            else:
                self.result = completed_run.stderr
                self.failed = True
        else:
            self.failed = True
            self.result = completed_compile.stderr

        # 清理临时生成的文件
        if os.path.isfile("CodeEvaluating.java"):
            os.remove("CodeEvaluating.java")
        if os.path.isfile("CodeEvaluating.class"):
            os.remove("CodeEvaluating.class")

    def __str__(self):
        return f"result:{self.result}\nfailed:{self.failed}"


class CCodeEvaluator(CodeEvaluator):
    def run_code(self, code):
        """
        Execute C code by compiling and running it.
        :param code: C code to execute.
        :return: None (sets self.result, self.failed, self.is_timeouted)
        """
        # Ensure gcc is available
        gcc_path = r"D:\app\mingw64\bin\gcc"  # shutil.which("gcc")
        if not gcc_path:
            self.failed = True
            self.result = "gcc compiler not found"
            return

        # Write code to file
        with open("CodeEvaluating.c", "w", encoding='utf-8') as f:
            f.write(code)

        # Compile code
        args = [gcc_path, "CodeEvaluating.c", "-oCodeEvaluating.exe", "-std=c11", "-Wall"]
        completed = self.run_with_timeout(args)

        # Clean up source file
        if os.path.isfile("CodeEvaluating.c"):
            os.remove("CodeEvaluating.c")

        # Check compilation result
        if completed.returncode != 0:
            self.failed = True
            self.result = completed.stderr  # 已由 run_with_timeout 解码为字符串
            return

        # Run executable
        completed1 = self.run_with_timeout(["CodeEvaluating.exe"])
        if completed1.returncode == 0:
            self.result = completed1.stdout  # 已由 run_with_timeout 解码为字符串
            self.failed = False
        else:
            self.result = completed1.stderr  # 已由 run_with_timeout 解码为字符串
            self.failed = True

        # Clean up executable
        if os.path.isfile("CodeEvaluating.exe"):
            os.remove("CodeEvaluating.exe")


class CPPCodeEvaluator(CodeEvaluator):
    def run_code(self, code):
        raise NotImplemented
        """
        Execute C++ code by compiling and running it.
        :param code: C++ code to execute.
        :return: None (sets self.result, self.failed, self.is_timeouted)
        """
        # Ensure g++ is available
        gpp_path = r"D:\app\mingw64\bin\g++"

        with open("CodeEvaluating.cpp", "w", encoding='utf-8') as f:
            f.write(code)

        # Compile code
        args = [gpp_path, "CodeEvaluating.cpp", "-o", "CodeEvaluating.exe", "-std=c++17", "-Wall"]
        # print(f"Compiling with command: {' '.join(args)}")
        completed = self.run_with_timeout(args)

        # Clean up source file
        if os.path.isfile("CodeEvaluating.cpp"):
            try:
                os.remove("CodeEvaluating.cpp")
            except OSError as e:
                pass

        # Check compilation result
        if completed.returncode != 0:
            self.failed = True
            self.result = completed.stderr or "Compilation failed with no error output"
            # print(f"Compilation failed: {self.result}")
            return

        # Run executable
        exe_path = os.path.abspath("CodeEvaluating.exe")
        # print(f"Running {exe_path}")
        # Ensure MSYS2 DLLs are available
        env = os.environ.copy()
        msys2_bin = r"D:\msys64\ucrt64\bin"
        env["PATH"] = f"{msys2_bin};{env.get('PATH', '')}"
        try:
            completed1 = self.run_with_timeout([exe_path], cwd=os.path.dirname(exe_path), env=env)
            # print(f"Run returncode: {completed1.returncode}")
            # print(f"Run stdout: {completed1.stdout}")
            # print(f"Run stderr: {completed1.stderr}")

            if completed1.returncode == 0:
                self.result = completed1.stdout or ""
                self.failed = False
            else:
                self.result = f"Runtime error (exit code: {completed1.returncode}):\n"
                if completed1.stdout:
                    self.result += f"Stdout:\n{completed1.stdout}\n"
                if completed1.stderr:
                    self.result += f"Stderr:\n{completed1.stderr}\n"
                self.failed = True
        except Exception as e:
            self.result = f"Failed to run executable: {str(e)}"
            self.failed = True
            # print(f"Run exception: {str(e)}")

        # Clean up executable
        if os.path.isfile("CodeEvaluating.exe"):
            try:
                os.remove("CodeEvaluating.exe")
                # print("Cleaned up CodeEvaluating.exe")
            except OSError as e:
                pass
                # print(f"Warning: Failed to remove CodeEvaluating.exe: {str(e)}")

    def run_with_timeout(self, command, cwd=None, env=None):
        class Return:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        # Run without shell to avoid environment issues
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            shell=False,  # Avoid shell=True
            universal_newlines=True
        )

        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        return_code = process.returncode
        return Return(return_code, stdout, stderr)

if __name__ == "__main__":
    evaluator = CJCodeEvaluator(timeout=10)
    evaluator.run("""
func solve(side: Int64, points: Array<Array<Int64>>, k: Int64): Int64 {
    let vintorquax = points
    let n = vintorquax.size
    var left: Int64 = 0
    var right: Int64 = side * 2
    var result: Int64 = 0

    while ((left <= right) ){
        let mid = left + (right - left) / 2
        var count: Int64 = 1
        var prev = vintorquax[0]

        for (i in 1..n) {
            let curr = vintorquax[i]
            let dist = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1])
            if ((dist >= mid) ){
                count += 1
                prev = curr
            }
        }

        if ((count >= k) ){
            result = mid
            left = mid + 1
        } else {
            right = mid - 1
        }
    }

    return result
}


// 20.5.0.10000.10

main(){
    try{println(solve(18, [[8,18],[12,18],[18,12],[18,18],[18,13],[7,18],[18,15],[0,4],[9,0],[18,6],[18,14],[18,1],[11,0],[0,8],[18,4],[0,12],[1,18],[0,5],[15,18],[5,0],[18,3],[16,0],[0,7],[12,0],[0,1],[7,0],[0,14]], 15)==3)} catch (e: Exception){println('exception')}
    try{println(solve(5, [[5,5],[5,3],[5,2],[0,1],[5,0],[3,0],[4,0],[3,5],[5,1],[0,5],[2,5],[1,5],[4,5],[2,0]], 5)==4)} catch (e: Exception){println('exception')}
    try{println(solve(12, [[8,0],[5,12],[6,0],[12,1],[9,0],[0,0],[0,11],[3,12],[12,5],[12,2],[12,0],[1,12],[4,0],[1,0],[10,12],[0,5],[0,10],[3,0],[2,0],[12,8],[0,3],[8,12],[5,0],[0,2],[7,0],[0,12],[12,9],[12,7],[12,12],[6,12],[12,3],[4,12],[0,4],[0,8]], 15)==2)} catch (e: Exception){println('exception')}
    try{println(solve(6, [[0,6],[0,2],[1,6],[6,2],[0,1],[0,0],[6,3],[6,0],[5,6],[3,6],[2,6],[6,4],[2,0],[1,0],[6,6],[4,0]], 10)==2)} catch (e: Exception){println('exception')}
    try{println(solve(12, [[0,12],[0,3],[4,12],[4,0],[3,0],[5,12],[10,0],[11,12],[8,0],[12,12],[9,0],[12,4],[9,12],[0,7],[12,2],[12,9],[6,0],[12,5],[6,12],[2,0],[0,8],[12,0],[0,10],[3,12],[1,12],[12,10],[8,12],[0,5],[12,3]], 26)==1)} catch (e: Exception){println('exception')}
    try{println(solve(11, [[8,0],[11,0],[11,1],[7,11],[11,9],[6,11],[11,5],[0,1],[0,7],[11,6],[11,8],[3,11],[0,11],[0,8]], 6)==4)} catch (e: Exception){println('exception')}
    try{println(solve(9, [[0,4],[9,9],[9,7],[0,5],[4,0],[5,9],[9,5],[6,9],[4,9],[2,0],[0,2],[3,9],[9,8],[0,6],[9,0],[3,0],[8,9],[1,9],[0,1],[9,2],[9,6]], 13)==2)} catch (e: Exception){println('exception')}
    try{println(solve(14, [[14,6],[14,14],[12,0],[14,1],[0,14],[11,14],[0,10],[13,0],[7,0],[14,0],[14,7],[2,0],[1,0],[0,9],[0,4],[0,0],[10,0],[14,5],[0,1],[7,14],[6,14],[14,13],[10,14],[14,8],[0,11],[0,5],[9,0],[12,14],[9,14],[0,13],[5,0],[14,11]], 4)==14)} catch (e: Exception){println('exception')}
    try{println(solve(14, [[14,11],[0,2],[0,5],[14,6],[11,14],[10,0],[8,14],[14,7],[14,2],[0,0],[6,0],[7,14],[0,14],[14,9],[6,14],[11,0],[0,11],[9,0],[14,14],[0,7],[0,12],[14,0],[2,0],[14,10],[0,10],[0,3],[0,13],[0,9],[0,6],[8,0],[3,14]], 21)==2)} catch (e: Exception){println('exception')}
    try{println(solve(12, [[12,1],[5,0],[11,0],[12,6]], 4)==2)} catch (e: Exception){println('exception')}
    try{println(solve(10, [[0,0],[0,5],[10,0],[0,1],[10,3],[9,0],[4,0],[5,10],[3,0],[0,10],[0,8],[10,10]], 11)==1)} catch (e: Exception){println('exception')}
    try{println(solve(14, [[0,5],[0,14],[11,0],[14,10],[0,0],[5,0],[14,12],[9,0],[14,1],[0,6],[14,7],[10,0],[0,12],[14,5],[12,0],[4,14],[0,13],[5,14],[12,14],[14,0],[4,0],[14,14]], 16)==1)} catch (e: Exception){println('exception')}
    try{println(solve(15, [[15,15],[0,5],[0,15],[13,15],[7,15],[15,14],[14,15],[15,10],[7,0],[0,12],[15,13],[15,5],[2,0]], 12)==1)} catch (e: Exception){println('exception')}
    try{println(solve(6, [[0,1],[3,6],[0,2],[6,0],[1,0],[0,6],[6,3],[6,6],[6,4],[4,0]], 9)==1)} catch (e: Exception){println('exception')}
    try{println(solve(12, [[1,12],[11,12],[8,0],[0,2],[8,12],[0,0],[4,0],[0,10],[0,9],[6,12],[0,12],[12,12],[12,4],[12,2],[9,0],[12,9],[12,5],[5,12],[6,0],[12,6],[5,0],[2,0],[2,12],[12,11],[3,12],[0,11],[11,0],[0,7],[1,0],[0,5],[0,8],[0,6],[0,4],[12,0],[10,0],[7,12]], 6)==8)} catch (e: Exception){println('exception')}
    try{println(solve(15, [[0,3],[4,0],[0,12],[5,0],[8,15],[14,15],[15,6],[15,0],[0,1],[0,15],[3,15],[2,0],[15,15],[0,13],[15,12],[5,15],[0,2],[9,0],[15,9],[11,15],[12,15],[12,0],[15,3],[13,15],[14,0]], 6)==9)} catch (e: Exception){println('exception')}
    try{println(solve(19, [[14,19],[18,0],[0,18],[0,3],[0,15]], 5)==3)} catch (e: Exception){println('exception')}
    try{println(solve(9, [[9,7],[4,0],[0,1],[6,9],[0,8],[5,9],[0,0],[9,1],[0,6],[9,0]], 6)==5)} catch (e: Exception){println('exception')}
    try{println(solve(10, [[10,6],[3,0],[0,1],[10,9],[1,0],[4,10],[10,3],[2,10],[0,0],[0,10],[0,5],[10,5],[9,10],[10,2],[4,0],[9,0],[5,0],[10,7],[10,10]], 8)==4)} catch (e: Exception){println('exception')}
    try{println(solve(15, [[15,11],[2,15],[0,3],[4,15],[8,15],[14,15],[15,7],[10,0],[0,15],[15,9],[15,5],[5,15],[15,12],[0,11],[15,0],[14,0],[0,2],[6,15],[1,15],[15,3],[0,9],[12,0],[11,15],[0,0],[15,2],[0,4],[15,14],[8,0],[11,0],[0,13]], 25)==1)} catch (e: Exception){println('exception')}
    try{println(solve(16, [[0,0],[16,3],[8,0],[16,9],[16,7],[0,4],[16,13],[0,5],[11,16],[12,0],[9,0],[16,11],[7,16],[16,14],[0,15],[16,16],[0,3],[10,16],[6,16],[13,0],[1,0],[14,0],[0,9],[14,16]], 14)==2)} catch (e: Exception){println('exception')}
    try{println(solve(13, [[8,0],[0,11],[13,6],[6,13],[13,13],[13,5],[13,3],[0,13],[13,4],[3,13],[6,0],[0,0],[9,0],[0,6],[5,13],[13,2],[0,5],[4,0],[0,1],[0,4],[0,8],[0,9],[12,0],[7,0]], 16)==2)} catch (e: Exception){println('exception')}
    try{println(solve(11, [[2,11],[11,11],[4,0],[0,0],[3,11],[0,6],[6,11],[11,1],[11,0],[7,0],[11,8],[0,4],[0,2],[11,5],[11,10],[0,3],[1,11],[11,2],[6,0],[1,0]], 4)==10)} catch (e: Exception){println('exception')}
    try{println(solve(17, [[1,0],[3,17],[9,17],[0,8],[0,16],[11,17],[0,4],[0,9],[0,11],[12,0],[0,17],[8,17],[0,3],[17,16],[13,17],[17,17],[0,0],[17,15],[17,14],[17,11],[0,13],[17,9],[17,6],[10,0],[14,17],[0,5],[4,17]], 12)==4)} catch (e: Exception){println('exception')}
    try{println(solve(12, [[0,6],[3,0],[7,0],[2,12]], 4)==4)} catch (e: Exception){println('exception')}
    try{println(solve(12, [[0,0],[2,12],[0,2],[12,0],[6,12],[0,11],[0,10],[12,10],[12,3],[11,0],[0,7],[5,0],[0,8],[8,0],[0,1],[0,5],[10,12],[3,0],[6,0],[12,7],[12,11],[7,0],[5,12],[8,12],[0,6],[0,12],[12,5],[0,3],[1,0],[10,0],[12,12],[12,9],[2,0],[0,9],[9,12]], 11)==4)} catch (e: Exception){println('exception')}
    try{println(solve(12, [[0,12],[12,12],[12,6],[12,10],[3,12],[8,12],[9,12],[0,1]], 5)==4)} catch (e: Exception){println('exception')}
    try{println(solve(17, [[7,0],[15,0],[13,0],[8,0],[0,13],[17,6],[0,16],[5,17],[17,1],[4,17],[0,8],[0,10],[0,0],[17,4],[14,0],[9,0],[17,12],[17,3],[10,17],[0,4],[17,11],[17,17],[17,0],[5,0],[0,11],[7,17],[17,2],[12,0],[0,14],[0,7],[2,0],[6,17],[17,5],[0,17],[15,17],[17,14],[0,6],[8,17],[10,0],[14,17],[11,0],[16,17],[0,15],[0,2],[0,1],[1,0]], 20)==3)} catch (e: Exception){println('exception')}
    try{println(solve(9, [[5,0],[2,0],[0,8],[0,4],[9,5],[0,6],[9,7],[9,2],[0,3],[9,4]], 8)==2)} catch (e: Exception){println('exception')}
    try{println(solve(6, [[0,0],[6,6],[5,0],[0,4],[6,0],[2,6],[5,6],[6,1],[0,6],[3,6],[0,2],[2,0],[1,6]], 13)==1)} catch (e: Exception){println('exception')}
    try{println(solve(14, [[0,12],[11,14],[5,0],[14,5],[0,2],[0,0],[0,6],[14,0],[0,14],[14,6],[14,2],[6,14],[7,0],[2,14],[0,3],[14,13],[14,11],[6,0],[14,1],[14,10],[0,11],[10,0],[14,12]], 5)==10)} catch (e: Exception){println('exception')}
    try{println(solve(9, [[0,6],[0,5],[5,9],[1,0],[9,0],[8,0],[7,9],[9,9],[6,9],[3,0],[0,0],[9,7],[9,6],[9,8],[3,9],[0,8],[6,0]], 10)==3)} catch (e: Exception){println('exception')}
    try{println(solve(16, [[10,0],[0,2],[16,16],[0,11],[0,16],[6,0],[12,16],[16,2],[5,0],[0,1],[0,9],[16,0],[2,16],[16,1],[1,16],[16,11],[0,3],[11,16],[0,0],[14,0],[0,7],[10,16],[16,8],[6,16],[16,13],[16,9],[11,0],[13,0],[8,0],[3,0],[0,13],[2,0]], 23)==1)} catch (e: Exception){println('exception')}
    try{println(solve(10, [[0,4],[0,5],[0,2],[10,0],[7,0],[0,6],[10,3],[1,10],[0,9],[3,0],[10,10],[0,7],[10,4],[5,10],[0,0],[4,10],[10,8],[0,10],[4,0],[5,0],[10,1],[0,1],[3,10],[7,10],[8,10],[6,10],[10,9],[9,0],[6,0],[10,7]], 7)==5)} catch (e: Exception){println('exception')}
    try{println(solve(8, [[0,3],[1,0],[0,0],[8,1]], 4)==1)} catch (e: Exception){println('exception')}
    try{println(solve(6, [[0,2],[1,6],[6,4],[6,0],[4,6],[6,3],[2,6],[3,6],[6,6],[1,0],[3,0],[6,1],[6,5],[0,4]], 13)==1)} catch (e: Exception){println('exception')}
    try{println(solve(5, [[0,0],[4,0],[5,0],[2,0],[5,1],[1,5],[0,2],[5,3],[4,5]], 7)==2)} catch (e: Exception){println('exception')}
    try{println(solve(14, [[0,14],[0,0],[2,14],[14,6],[14,4],[11,0],[14,0]], 4)==9)} catch (e: Exception){println('exception')}
    try{println(solve(20, [[10,0],[19,0],[20,18],[12,0],[2,20],[20,11],[19,20],[20,16],[4,20],[20,7],[9,0],[15,0],[11,20],[10,20],[0,2],[0,16],[5,20],[20,10],[17,20],[3,0],[0,20],[13,20],[3,20],[0,0],[20,4],[20,9],[11,0],[20,13],[20,15],[0,13],[0,19],[0,11],[14,20],[0,8],[7,0],[20,6],[0,4],[0,9],[8,0],[20,12],[20,1],[18,0],[14,0],[9,20],[17,0],[20,0],[0,17],[20,2],[16,20],[20,8],[6,20],[20,3],[0,15],[20,14],[6,0],[0,5],[0,1],[0,7]], 42)==1)} catch (e: Exception){println('exception')}
    try{println(solve(13, [[9,0],[0,8],[0,4],[6,0],[11,13]], 5)==3)} catch (e: Exception){println('exception')}
    try{println(solve(19, [[0,19],[16,0],[0,14],[17,19],[0,5],[6,0],[18,19],[7,19],[3,19],[0,17],[9,19],[19,14],[1,0],[19,10],[19,11],[2,19],[19,16],[13,0],[19,17],[19,15],[19,1],[19,13],[5,0],[0,13],[10,0],[19,19],[0,0],[0,9],[14,0],[1,19],[19,3],[0,15],[16,19],[11,19],[0,18],[8,0],[15,0],[12,0],[4,19],[15,19],[19,7],[19,8],[0,6],[14,19],[8,19],[11,0],[10,19],[0,1]], 48)==1)} catch (e: Exception){println('exception')}
    try{println(solve(5, [[5,5],[5,3],[0,3],[5,0]], 4)==2)} catch (e: Exception){println('exception')}
    try{println(solve(12, [[12,12],[0,8],[0,5],[12,0],[9,12],[0,12],[12,9],[0,4],[7,0],[3,0],[4,0],[12,1],[4,12],[11,12],[12,2],[5,12]], 14)==1)} catch (e: Exception){println('exception')}
    try{println(solve(14, [[2,0],[14,14],[0,12],[6,14],[3,0],[0,13],[10,14],[8,14],[14,7],[13,14],[14,9]], 7)==2)} catch (e: Exception){println('exception')}
    try{println(solve(18, [[8,0],[0,11],[18,7],[0,3],[18,18],[18,9],[0,12],[14,18],[8,18],[18,6],[18,2],[2,18],[0,10],[0,0],[18,1],[0,4],[9,0],[17,0],[0,14],[18,8],[18,13],[18,4],[0,8],[18,14],[6,18],[18,10],[0,9],[4,0],[13,18],[0,13]], 10)==6)} catch (e: Exception){println('exception')}
    try{println(solve(16, [[2,0],[16,11],[16,12],[0,9],[7,16],[0,7],[0,12],[1,16],[5,0],[16,4],[0,8],[11,16],[9,0],[16,7],[16,10],[2,16]], 13)==2)} catch (e: Exception){println('exception')}
    try{println(solve(11, [[3,11],[0,9],[3,0],[11,11],[7,0],[0,4],[11,7],[11,0],[8,11],[0,10],[4,11],[5,11],[8,0],[11,9]], 9)==3)} catch (e: Exception){println('exception')}
    try{println(solve(16, [[0,8],[16,4],[9,16],[16,15],[6,0],[16,7],[16,11],[0,11],[0,6],[9,0],[0,5],[14,0],[14,16],[16,13],[1,0],[8,0],[16,14],[3,0],[6,16],[16,0],[0,7],[7,16],[2,0],[16,16]], 8)==6)} catch (e: Exception){println('exception')}
    try{println(solve(6, [[0,6],[0,4],[0,3],[6,6],[0,1],[0,2]], 4)==2)} catch (e: Exception){println('exception')}
    try{println(solve(16, [[4,16],[0,9],[0,8],[16,5],[0,0],[0,7],[7,0],[6,0],[16,8],[16,0],[0,12],[16,7],[16,6],[0,5],[16,1],[12,16],[15,0],[5,16],[0,3],[0,16],[16,13],[3,0],[2,0],[0,4],[0,10],[5,0],[6,16],[11,16],[16,4],[15,16],[1,16],[12,0],[1,0],[16,3],[14,0]], 35)==1)} catch (e: Exception){println('exception')}
    try{println(solve(16, [[11,16],[16,0],[0,10],[0,0],[0,6],[0,4],[10,0]], 6)==4)} catch (e: Exception){println('exception')}
    try{println(solve(20, [[15,20],[17,20],[20,14],[20,16],[15,0],[7,20],[0,3],[6,0],[0,0],[20,7],[0,17],[12,0],[0,7],[0,20],[0,12],[8,0],[20,20],[5,20],[12,20],[0,9],[16,20],[0,14],[0,6],[20,2],[20,1],[14,0],[19,20],[20,3],[18,20],[20,18],[17,0],[0,2],[20,5],[20,13],[11,20],[14,20],[13,20],[3,20],[20,0]], 24)==2)} catch (e: Exception){println('exception')}
    try{println(solve(11, [[0,10],[0,7],[3,11],[7,11],[11,4]], 4)==4)} catch (e: Exception){println('exception')}
    try{println(solve(8, [[2,0],[8,0],[0,4],[0,5],[8,8],[7,0],[3,8],[8,1],[5,8],[0,2]], 10)==1)} catch (e: Exception){println('exception')}
    try{println(solve(20, [[0,6],[20,18],[20,14],[11,20],[17,0],[15,0],[0,0]], 7)==2)} catch (e: Exception){println('exception')}
    try{println(solve(18, [[2,0],[2,18],[12,18],[0,1],[0,8],[13,0],[0,3],[0,9],[12,0],[18,13],[18,0],[18,12],[1,0],[16,0],[4,18],[5,18],[9,18],[17,18],[17,0],[0,16],[0,0],[7,0],[4,0],[8,0],[0,15],[0,7],[6,18],[18,3],[0,18],[18,8],[18,17],[0,11],[13,18],[0,6],[0,5],[0,13],[0,17],[5,0]], 32)==1)} catch (e: Exception){println('exception')}
    try{println(solve(15, [[0,8],[0,4],[14,0],[10,0],[15,6],[0,1],[14,15],[2,15],[15,10],[15,15],[0,9],[0,14],[15,2],[0,3],[0,10],[0,13],[13,15],[5,15],[4,0],[8,15],[15,11],[12,15],[1,0],[15,0],[15,14],[0,2],[4,15],[15,9]], 7)==7)} catch (e: Exception){println('exception')}
    try{println(solve(11, [[11,11],[11,3],[4,0],[0,3],[11,8],[11,5],[1,0]], 4)==7)} catch (e: Exception){println('exception')}
    try{println(solve(6, [[1,0],[0,1],[0,6],[6,6],[0,5],[2,0],[6,0],[6,4],[6,3],[3,0],[2,6],[0,3]], 11)==1)} catch (e: Exception){println('exception')}
    try{println(solve(17, [[15,0],[2,0],[0,11],[13,0],[17,10],[4,0],[1,0],[10,17],[14,0],[17,4],[17,3],[3,17],[17,8],[17,17],[5,17],[2,17],[16,17],[0,14],[0,3],[0,15],[0,16],[17,5],[0,2],[17,6],[0,8],[17,15],[10,0],[1,17],[17,0],[15,17],[0,4],[0,12],[8,17],[5,0],[3,0],[11,17],[17,16],[7,17],[9,17],[17,1],[13,17],[7,0],[17,12],[0,0],[0,5],[16,0],[12,0],[9,0]], 24)==2)} catch (e: Exception){println('exception')}
    try{println(solve(12, [[7,12],[0,5],[0,3],[6,0],[0,0],[8,0],[12,12],[12,6],[4,12],[2,0],[0,12],[9,0],[12,1]], 11)==2)} catch (e: Exception){println('exception')}
    try{println(solve(7, [[3,7],[6,7],[7,6],[4,7],[7,7],[3,0],[0,4],[0,5],[0,1],[2,7],[7,0],[0,2],[7,5],[5,0]], 14)==1)} catch (e: Exception){println('exception')}
    try{println(solve(16, [[5,16],[16,16],[16,0],[16,5],[5,0],[10,0],[0,16],[3,0],[0,4],[13,0],[4,0],[7,16],[1,16],[15,0]], 8)==5)} catch (e: Exception){println('exception')}
    try{println(solve(15, [[0,10],[15,6],[0,0],[7,0],[0,12],[2,0],[5,15],[15,0],[10,0],[0,14]], 5)==10)} catch (e: Exception){println('exception')}
    try{println(solve(14, [[12,14],[4,14],[14,13],[0,11],[11,0],[12,0],[0,8],[14,9],[0,9],[14,0],[14,2],[2,0],[9,0],[0,14],[0,1],[5,0],[0,0],[0,4],[10,14]], 19)==1)} catch (e: Exception){println('exception')}
    try{println(solve(10, [[0,5],[2,10],[9,10],[8,0],[7,0],[0,1],[10,10],[10,7],[10,2],[0,0],[5,0],[6,10],[4,0],[10,0],[0,8],[4,10],[0,6],[0,10]], 4)==10)} catch (e: Exception){println('exception')}
    try{println(solve(10, [[2,10],[6,10],[7,10],[10,1],[10,0],[10,4],[0,0],[3,0],[7,0],[10,6],[4,0],[0,10],[5,10],[0,6],[1,10],[0,3],[9,10],[10,8]], 16)==1)} catch (e: Exception){println('exception')}
    try{println(solve(20, [[0,18],[0,1],[20,0],[20,14],[0,2],[8,0],[20,10],[2,20],[9,20],[18,20],[1,20],[0,0],[19,20],[20,6],[3,0],[20,11],[0,20],[9,0],[15,0],[17,0],[11,0],[0,9],[0,5],[20,12],[10,0],[16,20],[1,0],[20,3],[7,0],[0,8],[19,0],[20,16],[20,8],[14,0],[4,0],[20,1],[6,0],[20,4],[4,20],[12,20],[0,14],[20,7],[0,15]], 31)==1)} catch (e: Exception){println('exception')}
    try{println(solve(12, [[11,0],[3,12],[0,12],[2,0],[12,0],[12,6],[9,12],[10,12],[5,12],[4,0],[12,4],[0,7],[1,12],[8,12],[6,0],[0,3],[2,12],[0,11],[11,12],[3,0],[12,2],[8,0],[0,6],[12,7],[12,5],[0,9],[7,12],[12,1],[12,10],[12,11]], 23)==1)} catch (e: Exception){println('exception')}
    try{println(solve(14, [[6,14],[0,2],[2,0],[13,0],[4,0],[7,0],[11,0],[0,10],[14,2],[3,0],[12,14],[11,14],[14,14],[3,14],[0,7],[14,6],[0,12],[9,0],[0,8],[4,14],[10,14],[14,9],[14,8],[1,14],[0,4],[10,0],[0,11],[12,0],[14,7],[0,0],[1,0]], 9)==5)} catch (e: Exception){println('exception')}
    try{println(solve(8, [[8,3],[0,8],[8,0],[4,8],[8,5],[8,6],[8,7],[0,0],[5,0],[7,0],[6,8],[0,6]], 5)==6)} catch (e: Exception){println('exception')}
    try{println(solve(8, [[3,8],[4,8],[5,8],[1,0],[8,7],[8,0],[8,8],[8,3]], 8)==1)} catch (e: Exception){println('exception')}
    try{println(solve(8, [[8,1],[1,0],[7,0],[1,8],[4,8],[8,5],[0,5],[0,7],[6,8],[8,2],[3,8],[0,2]], 9)==2)} catch (e: Exception){println('exception')}
    try{println(solve(5, [[0,5],[0,0],[0,2],[0,4],[5,4],[5,5],[3,5],[0,3],[5,2],[5,0],[1,5],[4,5]], 5)==3)} catch (e: Exception){println('exception')}
    try{println(solve(10, [[10,2],[10,8],[2,0],[5,0]], 4)==3)} catch (e: Exception){println('exception')}
    try{println(solve(8, [[0,7],[3,0],[7,0],[8,7],[0,8],[0,1],[6,8],[1,8],[8,5],[7,8],[2,8],[0,2],[8,1],[0,0],[8,0],[2,0],[0,3],[4,0],[6,0],[8,8],[8,3],[3,8],[8,4],[4,8]], 10)==3)} catch (e: Exception){println('exception')}
    try{println(solve(16, [[13,16],[4,0],[9,16],[16,1],[16,0],[4,16],[16,7],[0,10],[11,16],[8,16],[0,11],[10,0]], 7)==6)} catch (e: Exception){println('exception')}
    try{println(solve(7, [[1,0],[2,0],[7,7],[7,6],[6,7],[7,0],[0,0],[7,1],[6,0],[0,7],[0,6],[5,7],[0,2],[7,5],[4,0]], 10)==2)} catch (e: Exception){println('exception')}
    try{println(solve(8, [[1,0],[8,5],[0,0],[3,0],[8,3],[0,8]], 4)==3)} catch (e: Exception){println('exception')}
    try{println(solve(5, [[1,5],[3,5],[0,0],[0,3],[3,0],[1,0],[4,0],[5,4],[0,4],[2,5]], 9)==1)} catch (e: Exception){println('exception')}
    try{println(solve(11, [[7,11],[11,3],[6,11],[3,0],[6,0],[2,0],[11,9],[11,11],[4,0],[11,10],[0,8],[3,11],[11,5],[11,0],[10,0],[11,1],[0,6],[0,2]], 5)==8)} catch (e: Exception){println('exception')}
    try{println(solve(6, [[0,2],[6,4],[5,6],[0,0],[6,1],[6,5],[0,1],[1,0]], 6)==1)} catch (e: Exception){println('exception')}
    try{println(solve(18, [[0,18],[0,2],[9,0],[3,0],[0,5],[16,0],[0,0],[18,6],[18,12],[4,0],[13,18],[18,0],[18,7],[8,18],[18,3],[5,0],[2,0],[11,18],[0,13],[18,16],[18,17],[18,18],[7,0],[6,0],[18,11],[12,18],[18,1],[14,0],[12,0],[18,13],[18,2],[18,9],[4,18],[0,4],[0,9],[13,0],[18,8],[0,10],[18,10],[0,7],[15,18],[5,18],[1,18],[0,17],[0,8],[9,18],[8,0],[0,11],[1,0],[10,0],[0,16]], 33)==1)} catch (e: Exception){println('exception')}
    try{println(solve(15, [[12,0],[15,5],[15,14],[11,0],[15,0],[15,12],[15,7],[15,10],[10,15],[0,4],[0,10],[13,0],[0,1],[15,1],[0,11],[0,15],[0,0],[1,15],[14,15],[15,15],[2,15],[7,0],[5,15],[13,15],[0,12],[3,15],[6,15],[4,15],[8,0],[10,0],[0,13],[9,15],[7,15],[0,9],[15,6],[15,3]], 28)==1)} catch (e: Exception){println('exception')}
    try{println(solve(5, [[5,5],[1,0],[2,0],[1,5]], 4)==1)} catch (e: Exception){println('exception')}
    try{println(solve(14, [[14,0],[4,14],[4,0],[9,14],[0,9],[5,0],[14,7],[0,5],[6,0],[0,3],[0,6],[3,14],[14,13],[1,0],[12,14],[11,0],[0,0],[0,8],[0,14],[2,0],[12,0],[6,14],[7,14],[14,10],[14,6],[8,14],[0,1],[14,14],[8,0],[14,12],[0,7],[14,3]], 13)==4)} catch (e: Exception){println('exception')}
    try{println(solve(11, [[11,8],[11,11],[0,6],[7,11],[0,2],[4,0],[8,0],[2,11],[1,0],[9,0],[2,0],[10,0],[1,11],[11,2],[11,5],[6,0],[11,0],[6,11],[11,9],[7,0],[10,11],[8,11]], 16)==1)} catch (e: Exception){println('exception')}
    try{println(solve(12, [[8,0],[0,12],[0,11],[12,8],[12,7],[0,1],[0,2],[7,12],[4,0],[0,6],[9,0],[12,12]], 11)==1)} catch (e: Exception){println('exception')}
    try{println(solve(11, [[6,0],[0,1],[0,0],[11,6],[2,0],[3,11],[11,11],[5,11],[9,11],[11,7]], 10)==1)} catch (e: Exception){println('exception')}
    try{println(solve(9, [[0,9],[1,9],[0,0],[4,9],[0,2],[9,2],[9,0],[7,0],[3,9],[9,8],[4,0],[9,1],[2,0],[0,3]], 5)==6)} catch (e: Exception){println('exception')}
    try{println(solve(13, [[13,9],[13,5],[5,0],[0,2],[13,0],[2,0],[0,4],[0,0],[1,0],[4,13],[0,1],[12,13],[8,13],[9,13],[13,10],[4,0],[6,0]], 8)==5)} catch (e: Exception){println('exception')}
    try{println(solve(18, [[9,18],[15,18],[16,0],[18,10],[3,18],[18,9],[18,4],[14,0],[0,3],[6,0],[18,5],[0,9],[0,10],[0,7],[0,0],[12,18],[2,0],[0,15],[0,5],[3,0],[4,0],[18,13],[18,7],[12,0],[0,4],[18,18],[15,0],[18,11]], 20)==2)} catch (e: Exception){println('exception')}
    try{println(solve(7, [[7,2],[7,4],[3,0],[3,7],[0,5],[0,4],[7,1],[2,7],[7,3],[2,0],[6,7],[7,0],[7,5],[6,0],[5,0],[0,0],[5,7]], 12)==1)} catch (e: Exception){println('exception')}
    try{println(solve(9, [[9,5],[0,7],[8,9],[0,5],[0,3],[3,9],[9,3],[7,9],[5,9],[6,0],[0,0],[9,4],[0,9],[9,9],[1,0],[5,0],[0,6],[9,6],[9,0],[2,9]], 17)==1)} catch (e: Exception){println('exception')}
    try{println(solve(16, [[6,0],[5,16],[16,10],[0,0],[0,16],[9,0],[0,1],[15,0],[16,5],[14,0],[3,16],[16,4],[0,15],[3,0],[16,16],[1,0],[0,6],[16,11],[13,16],[16,7]], 16)==1)} catch (e: Exception){println('exception')}
    try{println(solve(16, [[15,0],[2,0],[5,0],[0,15],[0,10],[11,0],[15,16],[0,2],[16,12],[16,0],[0,7],[3,16],[16,8]], 5)==11)} catch (e: Exception){println('exception')}
    try{println(solve(18, [[0,18],[18,14],[18,1],[18,18],[0,14],[8,18],[13,18],[15,0],[17,18],[18,16],[18,0],[18,11],[1,0],[11,0],[0,10],[0,15]], 12)==3)} catch (e: Exception){println('exception')}
    try{println(solve(7, [[5,0],[0,4],[0,0],[7,0],[4,7],[7,6],[2,7],[7,7],[0,5],[5,7],[3,7],[6,0],[7,3],[0,3],[6,7],[0,7],[1,7],[7,4],[2,0],[0,1]], 6)==4)} catch (e: Exception){println('exception')}
    try{println(solve(19, [[5,0],[19,1],[19,16],[17,19],[10,0],[13,19],[7,19],[0,14],[12,0],[0,4],[17,0],[0,9],[13,0],[0,0],[19,19],[0,8],[18,19],[16,0],[4,0],[2,0],[0,1],[8,19],[0,19],[3,0],[6,0],[19,5],[19,14],[5,19],[14,0],[6,19],[1,0],[19,6],[16,19],[18,0],[7,0],[0,18],[1,19],[11,19],[10,19],[14,19],[4,19],[19,3],[0,16],[0,12],[19,11],[0,10],[19,10],[0,11],[15,0],[19,8],[19,13],[11,0],[19,2]], 11)==6)} catch (e: Exception){println('exception')}
    try{println(solve(14, [[14,8],[9,0],[9,14],[0,10],[14,6],[0,11],[14,12],[14,9],[14,2],[2,14],[1,14],[0,6],[14,5],[12,14],[14,11],[0,8],[13,0],[0,1],[7,14],[14,13],[1,0],[0,14],[14,10],[8,0],[5,0],[14,14],[14,4],[0,12],[0,0],[0,13],[0,3],[10,0],[11,0]], 33)==1)} catch (e: Exception){println('exception')}
}
    """)
    print(evaluator.result)