mkdir clguetzli
echo f|xcopy ..\clguetzli\clguetzli.cl clguetzli\clguetzli.cl /y
echo f|xcopy ..\clguetzli\clguetzli.cl.h clguetzli\clguetzli.cl.h /y
echo f|xcopy ..\clguetzli\clguetzli.cu.ptx64 clguetzli\clguetzli.cu.ptx64 /y

echo f|xcopy ..\bin\x86_64\Release\guetzli.exe guetzli.exe /y