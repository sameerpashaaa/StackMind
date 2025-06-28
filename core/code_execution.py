import logging
import os
import sys
import subprocess
import tempfile
import json
import time
import re
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import shutil
import platform
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeExecution:
    def __init__(self, timeout: int = 30, memory_limit: int = 500,
                 allowed_modules: Optional[List[str]] = None,
                 blocked_modules: Optional[List[str]] = None,
                 safe_mode: bool = True):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.safe_mode = safe_mode

        self.default_allowed_modules = [
            'math', 'random', 'datetime', 'collections', 'itertools', 'functools',
            'string', 're', 'json', 'csv', 'time', 'copy', 'statistics', 'decimal',
            'fractions', 'heapq', 'bisect', 'array', 'enum', 'typing'
        ]

        self.default_blocked_modules = [
            'os', 'sys', 'subprocess', 'socket', 'shutil', 'pathlib', 'importlib',
            'pickle', 'shelve', 'marshal', 'dbm', 'sqlite3', 'multiprocessing',
            'threading', 'ctypes', 'builtins', 'traceback', 'pty', 'tty', 'pdb',
            'pwd', 'grp', 'crypt', 'spwd', 'signal', 'mmap', 'fcntl', 'resource',
            'posix', 'io', 'asyncio', 'tempfile', 'urllib', 'http', 'ftplib', 'poplib',
            'imaplib', 'nntplib', 'smtplib', 'telnetlib', 'uuid', 'binhex', 'base64',
            'ssl', 'requests', 'paramiko', 'pexpect', 'selenium', 'scrapy', 'twisted'
        ]

        self.allowed_modules = allowed_modules if allowed_modules is not None else self.default_allowed_modules
        self.blocked_modules = blocked_modules if blocked_modules is not None else self.default_blocked_modules

        self.node_available = self._check_node_available()

        self.os_type = platform.system().lower()

    def _check_node_available(self) -> bool:
        try:
            result = subprocess.run(['node', '--version'],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Node.js not found. JavaScript execution will not be available.")
            return False

    def _is_safe_code(self, code: str, language: str) -> Tuple[bool, str]:
        if not self.safe_mode:
            return True, ""

        if language.lower() == "python":
            import_pattern = r"(?:^|\n)\s*(?:import|from)\s+([\w\.]+)"
            imports = re.findall(import_pattern, code)

            for module in imports:
                base_module = module.split('.')[0]
                if base_module in self.blocked_modules:
                    return False, f"Blocked module: {base_module}"

                if self.allowed_modules and base_module not in self.allowed_modules:
                    return False, f"Module not in allowed list: {base_module}"

            dangerous_builtins = [
                r"\b__import__\b", r"\beval\b", r"\bexec\b", r"\bcompile\b",
                r"\bopen\b", r"\bfile\b", r"\binput\b", r"\braw_input\b"
            ]

            for pattern in dangerous_builtins:
                if re.search(pattern, code):
                    return False, f"Potentially dangerous function: {pattern.strip('\\b')}"

            dangerous_patterns = [
                r"\bos\.", r"\bsys\.", r"\bsubprocess\.", r"\bimportlib\.",
                r"\b__builtins__\b", r"\b__globals__\b", r"\b__dict__\b", r"\b__class__\b",
                r"\b__base__\b", r"\b__subclasses__\b", r"\b__mro__\b", r"\b__loader__\b"
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, code):
                    return False, f"Potentially dangerous pattern: {pattern}"

        elif language.lower() == "javascript":
            dangerous_js_patterns = [
                r"\brequire\b", r"\bprocess\.", r"\bchild_process\b", r"\bfs\.",
                r"\bpath\.", r"\bos\.", r"\bnet\.", r"\bhttp\.", r"\bhttps\.",
                r"\bcrypto\.", r"\bdns\.", r"\bvm\.", r"\bv8\.", r"\bBuffer\.",
                r"\beval\b", r"\bFunction\(", r"\bnew Function\b"
            ]

            for pattern in dangerous_js_patterns:
                if re.search(pattern, code):
                    return False, f"Potentially dangerous JavaScript pattern: {pattern}"

        elif language.lower() in ["shell", "bash", "powershell"]:
            dangerous_shell_patterns = [
                r"\brm\b", r"\bmkdir\b", r"\bchmod\b", r"\bchown\b", r"\bsudo\b",
                r"\bsu\b", r"\bdd\b", r"\bmv\b", r"\bcp\b", r"\bcat\b", r"\becho\b",
                r"\bgrep\b", r"\bawk\b", r"\bsed\b", r"\bcurl\b", r"\bwget\b",
                r"\bnc\b", r"\bnetcat\b", r"\btelnet\b", r"\bssh\b", r"\bftp\b",
                r"\bsftp\b", r"\bscp\b", r"\brsync\b", r"\bpython\b", r"\bperl\b",
                r"\bruby\b", r"\bphp\b", r"\bnode\b", r"\bnpm\b", r"\bpip\b",
                r"\bgem\b", r"\bcomposer\b", r"\bcargo\b", r"\bgo\b", r"\bjava\b",
                r"\bjavac\b", r"\bgcc\b", r"\bg\+\+\b", r"\bclang\b", r"\bmake\b",
                r"\bcmake\b", r"\bbuild\b", r"\binstall\b", r"\buninstall\b",
                r"\bservice\b", r"\bsystemctl\b", r"\binitctl\b", r"\blaunchctl\b",
                r"\bstart\b", r"\bstop\b", r"\brestart\b", r"\breload\b", r"\bkill\b",
                r"\bpkill\b", r"\bkillall\b", r"\btop\b", r"\bps\b", r"\bnetstat\b",
                r"\bifconfig\b", r"\bipconfig\b", r"\broute\b", r"\biptables\b",
                r"\bfirewall\b", r"\bufw\b", r"\bsetenforce\b", r"\bgetenforce\b",
                r"\bselinux\b", r"\bapparmor\b", r"\bchroot\b", r"\bjail\b",
                r"\bdocker\b", r"\bpodman\b", r"\bkubectl\b", r"\bhelm\b",
                r"\bopenssl\b", r"\bssh-keygen\b", r"\bgpg\b", r"\bcrypto\b",
                r"\bpasswd\b", r"\buseradd\b", r"\buserdel\b", r"\busermod\b",
                r"\bgroupadd\b", r"\bgroupdel\b", r"\bgroupmod\b", r"\bchage\b",
                r"\bshadow\b", r"\bpam\b", r"\bauthconfig\b", r"\bauthselect\b",
                r"\bldap\b", r"\bkerberos\b", r"\bsamba\b", r"\bwinbind\b",
                r"\bmount\b", r"\bumount\b", r"\bfsck\b", r"\bmkfs\b", r"\bfdisk\b",
                r"\bparted\b", r"\blvm\b", r"\bpvcreate\b", r"\bvgcreate\b",
                r"\blvcreate\b", r"\bpvremove\b", r"\bvgremove\b", r"\blvremove\b",
                r"\bswap\b", r"\bswapon\b", r"\bswapoff\b", r"\bfstab\b",
                r"\bcryptsetup\b", r"\bluks\b", r"\bdm-crypt\b", r"\becryptfs\b",
                r"\bencfs\b", r"\btruecrypt\b", r"\bveracrypt\b", r"\bbitlocker\b",
                r"\bfilevault\b", r"\bbackup\b", r"\brestore\b", r"\barchive\b",
                r"\bcompress\b", r"\bextract\b", r"\btar\b", r"\bzip\b", r"\bunzip\b",
                r"\bgzip\b", r"\bgunzip\b", r"\bbzip2\b", r"\bbunzip2\b", r"\bxz\b",
                r"\bunxz\b", r"\b7z\b", r"\brar\b", r"\bunrar\b", r"\bcpio\b",
                r"\bar\b", r"\barj\b", r"\blha\b", r"\blzh\b", r"\blzma\b",
                r"\blzo\b", r"\blz4\b", r"\bzstd\b", r"\bcompress\b", r"\buncompress\b",
                r"\bsqueeze\b", r"\bunsqueeze\b", r"\bpack\b", r"\bunpack\b",
                r"\bsplit\b", r"\bcsplit\b", r"\bjoin\b", r"\bpaste\b", r"\bcut\b",
                r"\bhead\b", r"\btail\b", r"\bsort\b", r"\buniq\b", r"\bwc\b",
                r"\bnl\b", r"\bod\b", r"\bhexdump\b", r"\bxxd\b", r"\bstrings\b",
                r"\bbase64\b", r"\buuencode\b", r"\buudecode\b", r"\bmd5sum\b",
                r"\bsha1sum\b", r"\bsha256sum\b", r"\bsha512sum\b", r"\bcksum\b",
                r"\bsum\b", r"\bcrc\b", r"\bcrc32\b", r"\bcrc64\b", r"\badler32\b",
                r"\bfind\b", r"\blocate\b", r"\bwhich\b", r"\bwhereis\b", r"\btype\b",
                r"\bcommand\b", r"\btime\b", r"\btimeout\b", r"\bnice\b", r"\brenice\b",
                r"\bionice\b", r"\bcpulimit\b", r"\bmemory\b", r"\bswap\b", r"\blimit\b",
                r"\bulimit\b", r"\brlimit\b", r"\bquota\b", r"\brepquota\b", r"\bedquota\b",
                r"\bsetquota\b", r"\bquotacheck\b", r"\bquotaon\b", r"\bquotaoff\b",
                r"\bcron\b", r"\bcrontab\b", r"\bat\b", r"\bbatch\b", r"\batrm\b",
                r"\batq\b", r"\batrun\b", r"\bsleep\b", r"\busleep\b", r"\bnanosleep\b",
                r"\bwait\b", r"\bwatch\b", r"\bnotify\b", r"\binotify\b", r"\bdnotify\b",
                r"\bfanotify\b", r"\bkqueue\b", r"\bepoll\b", r"\bpoll\b", r"\bselect\b",
                r"\bsignal\b", r"\bkill\b", r"\bpkill\b", r"\bkillall\b", r"\bkillall5\b",
                r"\bxkill\b", r"\bskill\b", r"\bsnice\b", r"\bstrace\b", r"\bltrace\b",
                r"\bdebug\b", r"\bgdb\b", r"\blldb\b", r"\bpdb\b", r"\bdbg\b", r"\bbreak\b",
                r"\bwatch\b", r"\bcore\b", r"\bcoredump\b", r"\bcrash\b", r"\bsegfault\b",
                r"\bsegmentation\b", r"\bfault\b", r"\berror\b", r"\bexception\b", r"\btrap\b",
                r"\binterrupt\b", r"\bsignal\b", r"\bhandler\b", r"\bcatch\b", r"\bthrow\b",
                r"\braise\b", r"\btry\b", r"\bexcept\b", r"\bfinally\b", r"\brescue\b",
                r"\bensure\b", r"\bbegin\b", r"\bend\b", r"\bif\b", r"\belse\b", r"\belif\b",
                r"\bfi\b", r"\bthen\b", r"\bcase\b", r"\besac\b", r"\bfor\b", r"\bwhile\b",
                r"\buntil\b", r"\bdo\b", r"\bdone\b", r"\bbreak\b", r"\bcontinue\b",
                r"\breturn\b", r"\bexit\b", r"\bquit\b", r"\babort\b", r"\bstop\b",
                r"\bhalt\b", r"\bshutdown\b", r"\breboot\b", r"\bpoweroff\b", r"\binit\b",
                r"\btelinit\b", r"\brunlevel\b", r"\bsysrq\b", r"\bmagic\b", r"\bsysreq\b",
                r"\bsysctl\b", r"\bmodprobe\b", r"\binsmod\b", r"\brmmod\b", r"\blsmod\b",
                r"\bmodinfo\b", r"\bdepmod\b", r"\bkernel\b", r"\bsysfs\b", r"\bprocfs\b",
                r"\btmpfs\b", r"\bdevfs\b", r"\budev\b", r"\bdbus\b", r"\bsystemd\b",
                r"\binit\b", r"\bupstart\b", r"\bopenrc\b", r"\brunit\b", r"\bs6\b",
                r"\bnosh\b", r"\bsysvinit\b", r"\binittab\b", r"\brc\b", r"\brc\.d\b",
                r"\binit\.d\b", r"\bservice\b", r"\bsystemctl\b", r"\binitctl\b",
                r"\blaunchctl\b", r"\blaunchd\b", r"\bplist\b", r"\bproperty\b",
                r"\blist\b", r"\bxml\b", r"\bjson\b", r"\byaml\b", r"\btoml\b",
                r"\bini\b", r"\bconf\b", r"\bconfig\b", r"\bcfg\b", r"\brc\b",
                r"\bprofile\b", r"\blogin\b", r"\blogout\b", r"\blogon\b", r"\blogoff\b",
                r"\bshell\b", r"\bbash\b", r"\bsh\b", r"\bcsh\b", r"\btcsh\b",
                r"\bksh\b", r"\bzsh\b", r"\bfish\b", r"\bcmd\b", r"\bcommand\b",
                r"\bpowershell\b", r"\bpwsh\b", r"\bps1\b", r"\bpsm1\b", r"\bpsd1\b",
                r"\bps1xml\b", r"\bpssc\b", r"\bpsrc\b", r"\bpsc1\b", r"\bcdxml\b",
                r"\bxsl\b", r"\bxslt\b", r"\bxml\b", r"\bhtml\b", r"\bhtm\b",
                r"\bcss\b", r"\bjs\b", r"\bjava\b", r"\bclass\b", r"\bjar\b",
                r"\bwar\b", r"\bear\b", r"\bejb\b", r"\bjsp\b", r"\bjspx\b",
                r"\bjsf\b", r"\bxhtml\b", r"\bphp\b", r"\bphp3\b", r"\bphp4\b",
                r"\bphp5\b", r"\bphp7\b", r"\bphp8\b", r"\bphtml\b", r"\bphar\b",
                r"\bpl\b", r"\bpm\b", r"\bt\b", r"\bpod\b", r"\bperl\b",
                r"\bpy\b", r"\bpyc\b", r"\bpyo\b", r"\bpyd\b", r"\bpyx\b",
                r"\bpyz\b", r"\bpywz\b", r"\bpyc\b", r"\bpyo\b", r"\bpyd\b",
                r"\bpyx\b", r"\bpyz\b", r"\bpywz\b", r"\bpython\b", r"\bpython2\b",
                r"\bpython3\b", r"\bpypy\b", r"\bpypy3\b", r"\bjython\b", r"\bipy\b",
                r"\bipython\b", r"\bipython2\b", r"\bipython3\b", r"\bjupyter\b",
                r"\bnb\b", r"\bipynb\b", r"\brb\b", r"\bruby\b", r"\birb\b",
                r"\berb\b", r"\bhaml\b", r"\bslim\b", r"\bjs\b", r"\bnode\b",
                r"\bnodejs\b", r"\bnpm\b", r"\byarn\b", r"\bpnpm\b", r"\bnpx\b",
                r"\bnvm\b", r"\bn\b", r"\bts\b", r"\btypescript\b", r"\btsc\b",
                r"\bgo\b", r"\bgolang\b", r"\brs\b", r"\brust\b", r"\bcargo\b",
                r"\bcs\b", r"\bcsharp\b", r"\bdotnet\b", r"\bnet\b", r"\bnetcore\b",
                r"\baspnet\b", r"\baspnetcore\b", r"\bmono\b", r"\broslyn\b",
                r"\bfs\b", r"\bfsharp\b", r"\bvb\b", r"\bvbnet\b", r"\bvisualbasic\b",
                r"\bc\b", r"\bcpp\b", r"\bc\+\+\b", r"\bclang\b", r"\bgcc\b",
                r"\bg\+\+\b", r"\bmingw\b", r"\bcygwin\b", r"\bmsvc\b", r"\bvc\+\+\b",
                r"\bvisualc\+\+\b", r"\bvisualstudio\b", r"\bvs\b", r"\bvscode\b",
                r"\bide\b", r"\beditor\b", r"\bcompiler\b", r"\binterpreter\b",
                r"\blinker\b", r"\bloader\b", r"\bdebugger\b", r"\bprofiler\b",
                r"\banalyzer\b", r"\blinter\b", r"\bformatter\b", r"\bbeautifier\b",
                r"\bminifier\b", r"\bobfuscator\b", r"\bdeobfuscator\b", r"\bdecompiler\b",
                r"\bdisassembler\b", r"\breverse\b", r"\bengineering\b", r"\bhack\b",
                r"\bcrack\b", r"\bexploit\b", r"\bvulnerability\b", r"\bsecurity\b",
                r"\bprivacy\b", r"\banonymity\b", r"\bencryption\b", r"\bdecryption\b",
                r"\bcryptography\b", r"\bsteganography\b", r"\bwatermark\b", r"\bdrm\b",
                r"\bcopyright\b", r"\blicense\b", r"\bpatent\b", r"\btrademark\b",
                r"\bintellectual\b", r"\bproperty\b", r"\bip\b", r"\blaw\b",
                r"\blegal\b", r"\billegal\b", r"\bcriminal\b", r"\bcivil\b",
                r"\bfederal\b", r"\bstate\b", r"\blocal\b", r"\binternational\b",
                r"\bglobal\b", r"\bworld\b", r"\buniverse\b", r"\bgalaxy\b",
                r"\bsolar\b", r"\bsystem\b", r"\bplanet\b", r"\bearth\b",
                r"\bmoon\b", r"\bsun\b", r"\bstar\b", r"\bcomet\b", r"\basteroid\b",
                r"\bmeteor\b", r"\bmeteor\b", r"\bmeteor\b", r"\bmeteor\b"
            ]

            for pattern in dangerous_shell_patterns:
                if re.search(pattern, code):
                    return False, f"Potentially dangerous shell pattern: {pattern}"

        return True, ""

    def execute_python(self, code: str, globals_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        is_safe, reason = self._is_safe_code(code, "python")
        if not is_safe:
            return {
                "success": False,
                "error": f"Code execution blocked: {reason}",
                "output": "",
                "result": None,
                "execution_time": 0
            }

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            if globals_dict is None:
                globals_dict = {}

            safe_builtins = {
                name: getattr(__builtins__, name)
                for name in dir(__builtins__)
                if name not in ['open', 'exec', 'eval', 'compile', '__import__']
            }

            locals_dict = {}

            stdout_buffer = StringIO()
            stderr_buffer = StringIO()

            start_time = time.time()
            result = None

            try:
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    if self.safe_mode:
                        cmd = [
                            sys.executable,
                            "-c",
                            f"import sys; sys.path.insert(0, '{os.path.dirname(temp_file_path)}'); "
                            f"exec(open('{temp_file_path}').read())"
                        ]

                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )

                        try:
                            stdout, stderr = process.communicate(timeout=self.timeout)
                            stdout_buffer.write(stdout)
                            stderr_buffer.write(stderr)

                            return_code = process.returncode
                            if return_code != 0:
                                raise Exception(f"Process exited with code {return_code}\n{stderr}")

                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.communicate()
                            raise TimeoutError(f"Code execution timed out after {self.timeout} seconds")

                    else:
                        exec(code, globals_dict, locals_dict)
                        result = locals_dict.get('result', None)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                stderr_buffer.write(error_msg)

            execution_time = time.time() - start_time

            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()

            try:
                os.unlink(temp_file_path)
            except:
                pass

            return {
                "success": stderr_output == "",
                "error": stderr_output,
                "output": stdout_output,
                "result": result,
                "execution_time": execution_time
            }

        except Exception as e:
            error_msg = f"Unexpected error during code execution: {str(e)}\n{traceback.format_exc()}"

            try:
                os.unlink(temp_file_path)
            except:
                pass

            return {
                "success": False,
                "error": error_msg,
                "output": "",
                "result": None,
                "execution_time": 0
            }

    def execute_javascript(self, code: str) -> Dict[str, Any]:
        if not self.node_available:
            return {
                "success": False,
                "error": "Node.js is not available. Cannot execute JavaScript code.",
                "output": "",
                "execution_time": 0
            }

        is_safe, reason = self._is_safe_code(code, "javascript")
        if not is_safe:
            return {
                "success": False,
                "error": f"Code execution blocked: {reason}",
                "output": "",
                "execution_time": 0
            }

        with tempfile.NamedTemporaryFile(suffix=".js", delete=False, mode="w") as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            start_time = time.time()

            process = subprocess.Popen(
                ["node", temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)

                execution_time = time.time() - start_time

                try:
                    os.unlink(temp_file_path)
                except:
                    pass

                return {
                    "success": process.returncode == 0,
                    "error": stderr,
                    "output": stdout,
                    "execution_time": execution_time
                }

            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()

                try:
                    os.unlink(temp_file_path)
                except:
                    pass

                return {
                    "success": False,
                    "error": f"Code execution timed out after {self.timeout} seconds",
                    "output": "",
                    "execution_time": self.timeout
                }

        except Exception as e:
            error_msg = f"Unexpected error during code execution: {str(e)}\n{traceback.format_exc()}"

            try:
                os.unlink(temp_file_path)
            except:
                pass

            return {
                "success": False,
                "error": error_msg,
                "output": "",
                "execution_time": 0
            }

    def execute_shell(self, command: str) -> Dict[str, Any]:
        is_safe, reason = self._is_safe_code(command, "shell")
        if not is_safe:
            return {
                "success": False,
                "error": f"Command execution blocked: {reason}",
                "output": "",
                "execution_time": 0
            }

        try:
            start_time = time.time()

            shell = True
            if self.os_type == "windows":
                command = f"powershell -Command \"{command}\""

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=shell,
                text=True
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)

                execution_time = time.time() - start_time

                return {
                    "success": process.returncode == 0,
                    "error": stderr,
                    "output": stdout,
                    "execution_time": execution_time
                }

            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()

                return {
                    "success": False,
                    "error": f"Command execution timed out after {self.timeout} seconds",
                    "output": "",
                    "execution_time": self.timeout
                }

        except Exception as e:
            error_msg = f"Unexpected error during command execution: {str(e)}\n{traceback.format_exc()}"

            return {
                "success": False,
                "error": error_msg,
                "output": "",
                "execution_time": 0
            }

    def execute_code(self, code: str, language: str, globals_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        language = language.lower()

        if language in ["python", "py"]:
            return self.execute_python(code, globals_dict)

        elif language in ["javascript", "js"]:
            return self.execute_javascript(code)

        elif language in ["shell", "bash", "powershell", "cmd"]:
            return self.execute_shell(code)

        else:
            return {
                "success": False,
                "error": f"Unsupported language: {language}",
                "output": "",
                "execution_time": 0
            }

    def format_result(self, result: Dict[str, Any]) -> str:
        output = ""

        if result["success"]:
            output += "✅ Execution successful\n"
        else:
            output += "❌ Execution failed\n"

        output += f"⏱️ Execution time: {result['execution_time']:.3f} seconds\n\n"

        if result["output"]:
            output += "📤 Output:\n"
            output += "```\n"
            output += result["output"]
            output += "\n```\n\n"

        if result["error"]:
            output += "⚠️ Error:\n"
            output += "```\n"
            output += result["error"]
            output += "\n```\n\n"

        if "result" in result and result["result"] is not None:
            output += "🔄 Result:\n"
            output += "```\n"
            output += str(result["result"])
            output += "\n```\n"

        return output

    def execute_and_format(self, code: str, language: str, globals_dict: Optional[Dict[str, Any]] = None) -> str:
        result = self.execute_code(code, language, globals_dict)
        return self.format_result(result)