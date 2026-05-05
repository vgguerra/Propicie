import pykinect2, os

# Corrigir time.clock()
path = os.path.join(os.path.dirname(pykinect2.__file__), 'PyKinectRuntime.py')
with open(path, 'r') as f:
    content = f.read()
content = content.replace('time.clock()', 'time.perf_counter()')
with open(path, 'w') as f:
    f.write(content)
print('PyKinectRuntime.py corrigido!')

# Corrigir PyKinectV2.py
path = os.path.join(os.path.dirname(pykinect2.__file__), 'PyKinectV2.py')
with open(path, 'r') as f:
    content = f.read()
content = content.replace('import numpy.distutils.system_info as sysinfo', '# import numpy.distutils.system_info as sysinfo')
content = content.replace("from comtypes import _check_version; _check_version('')", "# from comtypes import _check_version; _check_version('')")
content = content.replace('assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)', '# assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)')
with open(path, 'w') as f:
    f.write(content)
print('PyKinectV2.py corrigido!')