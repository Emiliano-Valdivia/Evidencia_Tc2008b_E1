using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentHub : MonoBehaviour
{
    public bool AgentTurn = false;
    private bool isRunning = false; 
    public int turno = 0;
    public int agente = 0;
    public GameObject hounter;
    public GameObject hounter2;
    public GameObject hounter3;
    public GameObject hounter4;
    public GameObject hounter5;
    public GameObject hounter6;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if(AgentTurn && !isRunning)
        {
            StartCoroutine(StartAgents());
        }
    }

    public IEnumerator StartAgents()
    {
        //Comienza una corrutina para cada agente, el número de agente se actualiza entre cada uno y
        //se regresa a 0 al final
        isRunning = true;
        yield return StartCoroutine(StartAgent1());
        agente ++;
        yield return StartCoroutine(StartAgent2());
        agente ++;
        yield return StartCoroutine(StartAgent3());
        agente ++;
        yield return StartCoroutine(StartAgent4());
        agente ++;
        yield return StartCoroutine(StartAgent5());
        agente ++;
        yield return StartCoroutine(StartAgent6());
        agente = 0;
        turno ++;
        GameObject.Find("TurnManager").GetComponent<WebClient>().TurnAdvance = true; //Indica al juego que debe recuperar más datos
        isRunning = false;
        AgentTurn = false;
    }

    //Corrutinas para cada agente, ejecuta su script interno y espera a que termine para pasar al siguiente.

    public IEnumerator StartAgent1()
    {
        GameObject.Find("hounter").GetComponent<AgentController>().turno = turno;
        GameObject.Find("hounter").GetComponent<AgentController>().agente = agente;
        GameObject.Find("hounter").GetComponent<AgentController>().enabled = true;
        yield return new WaitUntil(() => hounter.GetComponent<AgentController>().enabled == false);
    }

    public IEnumerator StartAgent2()
    {
        GameObject.Find("hounter (1)").GetComponent<AgentController>().turno = turno;
        GameObject.Find("hounter (1)").GetComponent<AgentController>().agente = agente;
        GameObject.Find("hounter (1)").GetComponent<AgentController>().enabled = true;
        yield return new WaitUntil(() => hounter2.GetComponent<AgentController>().enabled == false);
    }

    public IEnumerator StartAgent3()
    {
        GameObject.Find("hounter (2)").GetComponent<AgentController>().turno = turno;
        GameObject.Find("hounter (2)").GetComponent<AgentController>().agente = agente;
        GameObject.Find("hounter (2)").GetComponent<AgentController>().enabled = true;
        yield return new WaitUntil(() => hounter3.GetComponent<AgentController>().enabled == false);
    }

    public IEnumerator StartAgent4()
    {
        GameObject.Find("hounter (3)").GetComponent<AgentController>().turno = turno;
        GameObject.Find("hounter (3)").GetComponent<AgentController>().agente = agente;
        GameObject.Find("hounter (3)").GetComponent<AgentController>().enabled = true;
        yield return new WaitUntil(() => hounter4.GetComponent<AgentController>().enabled == false);
    }

    public IEnumerator StartAgent5()
    {
        GameObject.Find("hounter (4)").GetComponent<AgentController>().turno = turno;
        GameObject.Find("hounter (4)").GetComponent<AgentController>().agente = agente;
        GameObject.Find("hounter (4)").GetComponent<AgentController>().enabled = true;
        yield return new WaitUntil(() => hounter5.GetComponent<AgentController>().enabled == false);
    }

    public IEnumerator StartAgent6()
    {
        GameObject.Find("hounter (5)").GetComponent<AgentController>().turno = turno;
        GameObject.Find("hounter (5)").GetComponent<AgentController>().agente = agente;
        GameObject.Find("hounter (5)").GetComponent<AgentController>().enabled = true;
        yield return new WaitUntil(() => hounter6.GetComponent<AgentController>().enabled == false);
    }
}
