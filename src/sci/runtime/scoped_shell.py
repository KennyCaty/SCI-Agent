"""Restrict shell access with Landlock ABI 3 and seccomp (Linux x86-64)."""
import argparse
import ctypes as C
import errno
import os
from pathlib import Path
import platform
import shutil
import stat
import sys


class Ruleset(C.Structure):
    _fields_ = [('fs', C.c_uint64)]


class PathRule(C.Structure):
    _pack_ = 1
    _fields_ = [('access', C.c_uint64), ('parent_fd', C.c_int32)]


class Filter(C.Structure):
    _fields_ = [('code', C.c_ushort), ('jt', C.c_ubyte), ('jf', C.c_ubyte), ('k', C.c_uint32)]


class Program(C.Structure):
    _fields_ = [('length', C.c_ushort), ('filter', C.POINTER(Filter))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output-file',action='append',default=[])
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();workspace=a.workspace.resolve(strict=True)
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command:raise ValueError('Missing child command')
    if platform.machine()!='x86_64':raise RuntimeError('Shell sandbox requires Linux x86-64')
    libc=C.CDLL(None,use_errno=True)
    def check(result):
        if result<0:raise OSError(C.get_errno(),os.strerror(C.get_errno()))
        return result
    abi=check(libc.syscall(444,0,0,1))
    if abi<3:raise RuntimeError('Landlock ABI3 required; refusing unconfined execution')
    handled=(1<<15)-1;ro=1|4|8
    attr=Ruleset(handled);fd=check(libc.syscall(444,C.byref(attr),C.sizeof(attr),0))
    temp=workspace/'.tool_tmp';temp.mkdir(exist_ok=True)
    paths=[(workspace,ro),(temp,handled)]
    for path in ['/usr','/bin','/lib','/lib64',sys.prefix,
                 str(Path(shutil.which('node') or '/usr/bin/node').resolve().parent.parent),
                 '/etc/ld.so.cache','/etc/localtime','/etc/nsswitch.conf','/etc/passwd','/etc/group']:
        if Path(path).exists():paths.append((Path(path).resolve(),ro))
    for path in ['/dev/null','/dev/zero','/dev/urandom','/dev/random']:
        if Path(path).exists():paths.append((Path(path),2|4))
    ripgrep=Path(shutil.which('rg') or '/usr/bin/rg').resolve()
    if not ripgrep.is_file():raise RuntimeError('Install ripgrep (rg) on PATH')
    runtime_bin=ripgrep.parent
    paths.append((ripgrep,1|4))
    for path in a.output_file:
        paths.append((Path(path).resolve(strict=True),4))
    for path,access in paths:
        directory=stat.S_ISDIR(path.stat().st_mode)
        if not directory:access &= 1|2|4|(1<<14)
        item=os.open(path,os.O_PATH|os.O_CLOEXEC)
        rule=PathRule(access,item)
        try:check(libc.syscall(445,fd,1,C.byref(rule),0))
        finally:os.close(item)
    check(libc.prctl(38,1,0,0,0)) # PR_SET_NO_NEW_PRIVS
    check(libc.syscall(446,fd,0));os.close(fd)
    # Landlock ABI 3 has no network rules; seccomp blocks sockets and other architectures.
    instructions=[(0x20,0,0,4),(0x15,1,0,0xC000003E),(0x06,0,0,0x80000000),
                  (0x20,0,0,0),(0x35,0,1,0x40000000),(0x06,0,0,0x80000000),
                  (0x15,0,1,41),(0x06,0,0,0x00050000|errno.EPERM),
                  (0x06,0,0,0x7fff0000)]
    array=(Filter*len(instructions))(*(Filter(*x) for x in instructions))
    program=Program(len(instructions),array)
    check(libc.prctl(22,2,C.byref(program),0,0)) # PR_SET_SECCOMP, FILTER
    # Remove API credentials from the shell environment.
    env={k:v for k,v in os.environ.items() if k in {
        'PATH','LANG','LC_ALL','LC_CTYPE','TZ','HOME','USER','TERM',
        'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'}}
    env['PATH']=str(runtime_bin)+os.pathsep+env.get('PATH','/usr/bin:/bin')
    temp=workspace/'.tool_tmp';temp.mkdir(exist_ok=True);env['TMPDIR']=str(temp)
    os.chdir(workspace)
    os.execvpe(command[0],command,env)


if __name__=='__main__':main()
