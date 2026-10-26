import sys,tempfile,unittest,errno,os
from pathlib import Path
from unittest.mock import patch
from sci.runtime.workspace import acquire_file
class Check(unittest.TestCase):
 def test_link_or_identical_copy_and_no_silent_error(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);source=root/'source';source.write_bytes(b'exact corpus bytes\x00\xff')
   target=root/'link';acquire_file(source,target);self.assertEqual(source.stat().st_ino,target.stat().st_ino)
   for code in [errno.EPERM,errno.EACCES,errno.EXDEV]:
    target=root/str(code)
    with patch('sci.runtime.workspace.os.link',side_effect=OSError(code,'fixture')):acquire_file(source,target)
    self.assertEqual(target.read_bytes(),source.read_bytes())
    with patch('sci.runtime.workspace.os.link',side_effect=OSError(code,'fixture')):
     with self.assertRaises(FileExistsError):acquire_file(source,target)
   with patch('sci.runtime.workspace.os.link',side_effect=OSError(errno.ENOSPC,'fixture')):
    with self.assertRaises(OSError):acquire_file(source,root/'full')
   self.assertFalse((root/'full').exists())
if __name__=='__main__':unittest.main()
