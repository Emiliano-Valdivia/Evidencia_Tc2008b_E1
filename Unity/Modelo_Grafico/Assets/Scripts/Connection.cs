// TC2008B Modelaci�n de Sistemas Multiagentes con gr�ficas computacionales
// C# client to interact with Python server via POST
// Sergio Ruiz-Loza, Ph.D. March 2021
using Newtonsoft.Json;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Networking;

public class WebClient : MonoBehaviour
{
    //Recupera datos del servidor y los almacena en objetos
    public Turn Board;
    public AgentTurn Agents;
    public bool TurnAdvance = true;
    // IEnumerator - yield return
    IEnumerator SendData(string data)
    {
        TurnAdvance = false;
        WWWForm form = new WWWForm();
        form.AddField("bundle", "the data");
        string url = "http://localhost:8585";
        using (UnityWebRequest www = UnityWebRequest.Post(url, form))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(data);
            www.uploadHandler = (UploadHandler)new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
            //www.SetRequestHeader("Content-Type", "text/html");
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();          // Talk to Python
            if (www.isNetworkError || www.isHttpError)
            {
                Debug.Log(www.error);
            }
            else
            {
                string responseText = www.downloadHandler.text;
                Debug.Log(www.downloadHandler.text);    // Answer from Python
                string[] jsonParts = responseText.Split('\n'); //Al recibir varios json, los divide y almacena en un arreglo
                Debug.Log(jsonParts[0]);
                Debug.Log(jsonParts[1]);

                //Cada parte del arreglo de json se almacena para representar diferente información

                Agents = JsonConvert.DeserializeObject<AgentTurn>(jsonParts[0]);
                Board = JsonConvert.DeserializeObject<Turn>(jsonParts[1]);
                GameObject.Find("BoardManager").GetComponent<Tablero>().BoardTurn = true; //Indica al tablero que es su turno
            }
        }

    }


    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if(TurnAdvance)
        {
            Vector3 fakePos = new Vector3(3.44f, 0, -15.707f);
            string json = EditorJsonUtility.ToJson(fakePos);
            StartCoroutine(SendData(json));
        }
    }
}