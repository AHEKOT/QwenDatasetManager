$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$AppPath = Join-Path $ProjectRoot 'app.py'

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Trainer environment not found: $PythonPath. Run install.cmd first."
}

if (-not (Test-Path -LiteralPath $AppPath -PathType Leaf)) {
    throw "Application entry point not found: $AppPath"
}

# Create the server suspended, attach it to a kill-on-close Windows Job Object,
# and only then let it execute. Starting suspended avoids the venv launcher
# spawning the real Python interpreter before it belongs to the same job.
if (-not ('QdmConsoleJob' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public sealed class QdmJobProcess
{
    public IntPtr JobHandle;
    public IntPtr ProcessHandle;
    public int ProcessId;
}

public static class QdmConsoleJob
{
    private const uint CREATE_SUSPENDED = 0x00000004;
    private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    private const uint INFINITE = 0xFFFFFFFF;

    private enum JobObjectInfoType
    {
        ExtendedLimitInformation = 9
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct STARTUPINFO
    {
        public int cb;
        public string lpReserved;
        public string lpDesktop;
        public string lpTitle;
        public uint dwX;
        public uint dwY;
        public uint dwXSize;
        public uint dwYSize;
        public uint dwXCountChars;
        public uint dwYCountChars;
        public uint dwFillAttribute;
        public uint dwFlags;
        public ushort wShowWindow;
        public ushort cbReserved2;
        public IntPtr lpReserved2;
        public IntPtr hStdInput;
        public IntPtr hStdOutput;
        public IntPtr hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PROCESS_INFORMATION
    {
        public IntPtr hProcess;
        public IntPtr hThread;
        public uint dwProcessId;
        public uint dwThreadId;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr jobAttributes, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(
        IntPtr job,
        JobObjectInfoType infoType,
        IntPtr info,
        uint infoLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateProcess(
        string applicationName,
        StringBuilder commandLine,
        IntPtr processAttributes,
        IntPtr threadAttributes,
        bool inheritHandles,
        uint creationFlags,
        IntPtr environment,
        string currentDirectory,
        ref STARTUPINFO startupInfo,
        out PROCESS_INFORMATION processInformation);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint ResumeThread(IntPtr thread);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateProcess(IntPtr process, uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);

    public static QdmJobProcess Start(string executable, string script, string workingDirectory)
    {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero)
            throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateJobObject failed");

        IntPtr infoPointer = IntPtr.Zero;
        PROCESS_INFORMATION processInfo = new PROCESS_INFORMATION();
        bool processCreated = false;

        try
        {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION info =
                new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            int infoSize = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
            infoPointer = Marshal.AllocHGlobal(infoSize);
            Marshal.StructureToPtr(info, infoPointer, false);

            if (!SetInformationJobObject(
                    job,
                    JobObjectInfoType.ExtendedLimitInformation,
                    infoPointer,
                    (uint)infoSize))
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "SetInformationJobObject failed");

            STARTUPINFO startupInfo = new STARTUPINFO();
            startupInfo.cb = Marshal.SizeOf(typeof(STARTUPINFO));
            StringBuilder commandLine = new StringBuilder(
                "\"" + executable + "\" \"" + script + "\"");

            if (!CreateProcess(
                    executable,
                    commandLine,
                    IntPtr.Zero,
                    IntPtr.Zero,
                    true,
                    CREATE_SUSPENDED,
                    IntPtr.Zero,
                    workingDirectory,
                    ref startupInfo,
                    out processInfo))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateProcess failed");

            processCreated = true;
            if (!AssignProcessToJobObject(job, processInfo.hProcess))
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "AssignProcessToJobObject failed");

            if (ResumeThread(processInfo.hThread) == UInt32.MaxValue)
                throw new Win32Exception(Marshal.GetLastWin32Error(), "ResumeThread failed");

            CloseHandle(processInfo.hThread);
            processInfo.hThread = IntPtr.Zero;

            return new QdmJobProcess {
                JobHandle = job,
                ProcessHandle = processInfo.hProcess,
                ProcessId = (int)processInfo.dwProcessId
            };
        }
        catch
        {
            if (processCreated && processInfo.hProcess != IntPtr.Zero)
                TerminateProcess(processInfo.hProcess, 1);
            if (processInfo.hThread != IntPtr.Zero)
                CloseHandle(processInfo.hThread);
            if (processInfo.hProcess != IntPtr.Zero)
                CloseHandle(processInfo.hProcess);
            CloseHandle(job);
            throw;
        }
        finally
        {
            if (infoPointer != IntPtr.Zero)
                Marshal.FreeHGlobal(infoPointer);
        }
    }

    public static void Close(QdmJobProcess state)
    {
        if (state == null)
            return;
        if (state.JobHandle != IntPtr.Zero)
        {
            CloseHandle(state.JobHandle);
            state.JobHandle = IntPtr.Zero;
        }
        if (state.ProcessHandle != IntPtr.Zero)
        {
            CloseHandle(state.ProcessHandle);
            state.ProcessHandle = IntPtr.Zero;
        }
    }
}
'@
}

$JobProcess = $null
$ServerProcess = $null
$ExitCode = 1

try {
    $JobProcess = [QdmConsoleJob]::Start($PythonPath, $AppPath, $ProjectRoot)
    $ServerProcess = [System.Diagnostics.Process]::GetProcessById($JobProcess.ProcessId)
    Write-Host "Server process $($JobProcess.ProcessId) is attached to this console."
    Write-Host 'Closing this console will stop the complete server process tree.'
    $ServerProcess.WaitForExit()
    $ExitCode = $ServerProcess.ExitCode
}
finally {
    if ($ServerProcess) {
        $ServerProcess.Dispose()
    }
    [QdmConsoleJob]::Close($JobProcess)
}

exit $ExitCode
