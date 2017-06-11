mkdir clguetzli
echo f|copy ..\clguetzli\clguetzli.cl clguetzli\clguetzli.cl /y
echo f|copy ..\clguetzli\clguetzli.cl.h clguetzli\clguetzli.cl.h /y
echo f|copy ..\clguetzli\clguetzli.cu.ptx64 clguetzli\clguetzli.cu.ptx64 /y

echo f|copy ..\bin\x86_64\Release\guetzli.exe guetzli.exe /y