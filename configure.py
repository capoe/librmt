#! /usr/bin/env python
import os

cwd = os.getcwd()
cwd = os.path.abspath(cwd)

#pypath = os.environ['PYTHONPATH']
#if not cwd in pypath:
#    os.environ['PYTHONPATH'] = pypath + ':%s' % cwd
#    print "Extended python path:", os.environ['PYTHONPATH']

ifs = open('LIBRMTRC.in', 'r')
ofs = open('LIBRMTRC', 'w')
for ln in ifs.readlines():
    ln = ln.replace('@CMAKE_INSTALL_PREFIX@', cwd)
    ofs.write(ln)
ofs.close()
ifs.close()
