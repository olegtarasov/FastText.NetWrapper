using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using FastText.NetWrapper.Logging;
using FastText.NetWrapper.Properties;

namespace FastText.NetWrapper
{
    public partial class FastTextWrapper
    {
        private static readonly object _resourceLocker = new object();
        private static bool _libLoaded = false;

        private static void LoadNativeLibrary()
        {
            if (_libLoaded)
            {
                return;
            }

            lock (_resourceLocker)
            {
                if (_libLoaded)
                {
                    return;
                }

                string dir = UnpackResources();
                string lib = Path.Combine(dir, FastTextDll);

                LoadLibrary(lib);

                _libLoaded = true;
            }
        }

        private static void LoadLibrary(string path)
        {
            _log.Info($"Directly loading {path}...");
            var result = LoadLibraryEx(path, IntPtr.Zero, LoadLibraryFlags.LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LoadLibraryFlags.LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LoadLibraryFlags.LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LoadLibraryFlags.LOAD_LIBRARY_SEARCH_SYSTEM32 | LoadLibraryFlags.LOAD_LIBRARY_SEARCH_USER_DIRS);
            _log.Info(result == IntPtr.Zero ? "FAILED!" : "Success");
        }

        private static string UnpackResources()
        {
            string curDir;
            var ass = Assembly.GetExecutingAssembly().Location;
            if (string.IsNullOrEmpty(ass))
            {
                curDir = Environment.CurrentDirectory;
            }
            else
            {
                curDir = Path.GetDirectoryName(ass);
            }

            _log.Info($"Unpacking native libs to {curDir}");

            UnpackFile(curDir, "FastText.dll", Resources.FastText);
            
            return curDir;
        }

        private static void UnpackFile(string curDir, string fileName, byte[] bytes)
        {
            var path = !string.IsNullOrEmpty(curDir) ? Path.Combine(curDir, fileName) : fileName;

            _log.Info($"Unpacking {fileName}.");

            File.WriteAllBytes(path, bytes);
        }

        #region LoadLibraryEx

        [System.Flags]
        private enum LoadLibraryFlags : uint
        {
            DONT_RESOLVE_DLL_REFERENCES = 0x00000001,
            LOAD_IGNORE_CODE_AUTHZ_LEVEL = 0x00000010,
            LOAD_LIBRARY_AS_DATAFILE = 0x00000002,
            LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE = 0x00000040,
            LOAD_LIBRARY_AS_IMAGE_RESOURCE = 0x00000020,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR = 0x00000200,
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100,
            LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800,
            LOAD_LIBRARY_SEARCH_USER_DIRS = 0x00000400,
            LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr LoadLibraryEx(string lpFileName, IntPtr hReservedNull, LoadLibraryFlags dwFlags);

        #endregion
    }
}