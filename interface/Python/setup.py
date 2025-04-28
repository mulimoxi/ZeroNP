from setuptools import setup
import platform

if platform.system() == "Linux":
    zeronp_lib = "../../build/libzeronp.so"
elif platform.system() == "Windows":
    # zeronp_lib = "../build/Debug/zeronp.dll"
    zeronp_lib = "../../build/libzeronp.dll"
else:
    # raise Exception("%s Platform not supported yet" % platform.system())
    print(platform.system())
    zeronp_lib = "../../build/libzeronp.dylib"

setup(
    name="pyzeronp",
    version="1.0",
    author="LLT",
    py_modules=["pyzeronp", "ZERONP_CONST"],
    data_files=[zeronp_lib],
)
