# java-cuda-array
Call cuda array ops and intrinsic function from java via panama FFM

# Compile Cuda in Windows OS
1) Enter x86_64 Cross Tool Command (Visual Studio MSVC Compiler)
2) `cd cusrc`
3) Type `nvcc -shared Vector.cu -o Vector.dll`
4) Move `Vector.dll` next to Main.java, main.rb, main.py

# Java API
1) `javac -cp . Main.java` and `java -cp . Main`

# Ruby API
1) `ruby main.rb`

# Python API
1) `py main.py`
