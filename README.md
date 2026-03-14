# java-cuda-array
Call cuda array ops and intrinsic function from java via panama FFM

Windows Usage
1) Enter x86_64 Cross Tool Command (Visual Studio MSVC Compiler)
2) `cd cusrc`
3) Type `nvcc -shared Vector.cu -o Vector.dll`
4) Move `Vector.dll` next to Main.java
5) `javac -cp . Main.java` and `java -cp . Main`
