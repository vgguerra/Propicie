using System.Diagnostics;
using UnityEngine;

public class PythonRunner : MonoBehaviour
{
    public void StartPythonScript()
    {
        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = "python"; 
        psi.Arguments = "sit_and_reach_holistic_victor.py";  
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.CreateNoWindow = true;

        Process process = new Process();
        process.StartInfo = psi;
        process.Start();

        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();

        UnityEngine.Debug.Log("Saída do Python: " + output);
        if (!string.IsNullOrEmpty(error))
        {
            UnityEngine.Debug.LogError("Erro no Python: " + error);
        }

        process.WaitForExit();
    }

    void Start()
    {
        StartPythonScript();  
    }
}
