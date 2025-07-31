using System.IO;
using UnityEngine;

public class PoseLogger : MonoBehaviour
{
    public Transform mainCameraTransform;         // Unity MainCamera
    public Transform simulationCameraTransform;   // Unity SimulationCamera

    // Harici veriler — dışardan set edilecek

    private string filePath;
    private StreamWriter writer;
    private float logInterval = 0.1f;
    private float timeSinceLastLog = 0f;

    void Start()
    {
        try
        {
            filePath = Path.Combine(Application.temporaryCachePath, "FullPoseLog.csv");
            writer = new StreamWriter(filePath, false);

            // Başlık satırı
            writer.WriteLine("Time," +
                "MainCamX,MainCamY,MainCamZ,MainCamRotX,MainCamRotY,MainCamRotZ,MainCamRotW," +
                "SimCamX,SimCamY,SimCamZ,SimCamRotX,SimCamRotY,SimCamRotZ,SimCamRotW");

            Debug.Log("Pose logger started: " + filePath);
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Logger başlatılamadı: " + ex.Message);
        }
    }

    void Update()
    {
        timeSinceLastLog += Time.deltaTime;

        if (timeSinceLastLog >= logInterval)
        {
            LogData();
            timeSinceLastLog = 0f;
        }
    }

    void LogData()
    {
        if (mainCameraTransform == null || simulationCameraTransform == null)
        {
            Debug.LogWarning("Logger: Kamera referansları eksik!");
            return;
        }

        Vector3 mcPos = mainCameraTransform.position;
        Quaternion mcRot = mainCameraTransform.rotation;

        Vector3 simPos = simulationCameraTransform.position;
        Quaternion simRot = simulationCameraTransform.rotation;

        float time = Time.time;

        string line = string.Format("{0:F3}," +
                                    "{1:F4},{2:F4},{3:F4},{4:F4},{5:F4},{6:F4},{7:F4}," +
                                    "{8:F4},{9:F4},{10:F4},{11:F4},{12:F4},{13:F4},{14:F4}",
            time,
            mcPos.x, mcPos.y, mcPos.z, mcRot.x, mcRot.y, mcRot.z, mcRot.w,
            simPos.x, simPos.y, simPos.z, simRot.x, simRot.y, simRot.z, simRot.w
        );


        try
        {
            writer.WriteLine(line);
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Logger yazma hatası: " + ex.Message);
        }
    }

    private void OnApplicationQuit()
    {
        if (writer != null)
        {
            writer.Close();
            Debug.Log("Kayıt durduruldu ve dosya kapatıldı.");
        }
    }
}
