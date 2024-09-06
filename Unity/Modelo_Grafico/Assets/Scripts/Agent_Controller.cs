using UnityEngine;
using System.Collections.Generic;
using System.Collections;

public class AgentController : MonoBehaviour
{
    public AgentTurn agentData;
    public GameObject hounter;
    public float moveSpeed = 0.5f;
    public int turno = 0;
    public int agente = 0;

    // Clips de animacion
    public AnimationClip idleAnimation;
    public AnimationClip walkAnimation;

    private Animation anim; // Referencia al componente Animation
    private Vector3 targetPosition;
    public string clave => $"({turno}, {agente})";

    void Start()
    {
        // Inicializar la animacion
        anim = hounter.GetComponent<Animation>();

        // Asignar los clips al componente Animation
        anim.AddClip(idleAnimation, "idle");
        anim.AddClip(walkAnimation, "walk");

        // Reproducir la animacion de "idle" al inicio
        anim.Play("idle");

        // Inicializar la posicion objetivo
        targetPosition = hounter.transform.position;
    }

    void Update()
    {
        StartCoroutine(ExecuteAgentRoutine(clave));
    }

    public void updateAgent() //Actualiza los datos del agente a partir de lo recibido desde el servidor
    {
        agentData = GameObject.Find("TurnManager").GetComponent<WebClient>().Agents;
    }

    public IEnumerator ApplyAgentData(string key) //Corrutina que mueve al agente a la posicion actualizada y ejecuta sus animaciones
    {
        updateAgent();
        
        if (agentData != null)
        {
            List<int> positionList = agentData.Pos[key];
            if (positionList != null && positionList.Count == 2)
            {
                targetPosition = new Vector3(3.5f + positionList[1] * 3, 0.1f, 14.5f - positionList[0] * 3);
                
                // Iniciar la animación de caminar
                anim.CrossFade("walk");
                while (Vector3.Distance(hounter.transform.position, targetPosition) > 0.001f)
                {
                    hounter.transform.position = Vector3.MoveTowards(hounter.transform.position, targetPosition, moveSpeed * Time.deltaTime);
                    yield return null; // Esperar al siguiente frame
                }

                // Una vez alcanzada la posición, cambiar a la animación de idle
                anim.CrossFade("idle");
                yield return new WaitForSeconds(idleAnimation.length); // Esperar a que termine la animación de idle
            }
            else
            {
                Debug.LogWarning("La lista de posiciones no tiene suficientes elementos o es nula.");
            }
        }
        else
        {
            Debug.LogWarning("AgentTurn no está asignado.");
        }
    }

    public IEnumerator ExecuteAgentRoutine(string key)
    {
        yield return StartCoroutine(ApplyAgentData(key));
        // Esperar un tiempo entre animaciones
        yield return new WaitForSeconds(0.3f);
        this.enabled = false;
        // Se desactiva el script para dejar al agente quieto hasta que su turno llegue
    }
}
