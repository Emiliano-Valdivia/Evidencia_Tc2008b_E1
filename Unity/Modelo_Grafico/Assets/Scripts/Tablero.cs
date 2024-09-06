using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEditor;
using System.Linq;

public class Tablero : MonoBehaviour
{
    //Se preparan objetos para recibir la información proveniente del servidor e  interpretarla
    public GameObject vamp;
    public GameObject bats;
    public GameObject interest;
    public GameObject interestf;
    public int turno = 0;
    public bool BoardTurn = false;
    private bool isRunning = false;
    public int dead = 0;
    public int rescued = 0;
    public int damage = 0;
    List<List<int>> fuego = new List<List<int>>();
    List<List<int>> humo = new List<List<int>>();
    List<List<int>> victims = new List<List<int>>();
    List<List<int>> falsea = new List<List<int>>();
    List<List<int>> pastFuego = new List<List<int>>();
    List<List<int>> pastHumo = new List<List<int>>();
    Turn Board;
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if(BoardTurn && !isRunning)
        {
            StartCoroutine(GenerateEvents());
        }
    }

    public void UpdateMap() //Se busca la información recibida desde el servidor y se pone en un objeto
    {
        Board = GameObject.Find("TurnManager").GetComponent<WebClient>().Board;
    }

    private bool AreListsEqual(List<int> list1, List<int> list2) //Compara los elementos de 2 listas
    {
        if (list1.Count != list2.Count) return false;
        for (int i = 0; i < list1.Count; i++)
        {
            if (list1[i] != list2[i]) return false;
        }
        return true;
    }

    public IEnumerator GenerateEvents()
    {
        BoardTurn = false; //Cambia la variable del turno a falso para que solo se ejecute 1 vez en el step
        isRunning = true; //Cambia una variable a verdadero con el mismo propósito de la variable anterior
        UpdateMap();
        //Se inician corrutinas para actualizar cada aspecto del tablero
        yield return StartCoroutine(CreateSmoke(turno));
        pastHumo = humo.Select(x => new List<int>(x)).ToList();
        yield return StartCoroutine(CreateFire(turno));
        pastFuego = fuego.Select(x => new List<int>(x)).ToList();
        yield return StartCoroutine(CreateVictims(turno));
        yield return StartCoroutine(CreateFAs(turno));
        yield return GetDead(turno);
        yield return GetRescued(turno);
        yield return GetDP(turno);
        turno += 1;
        GameObject.Find("AgentManager").GetComponent<AgentHub>().AgentTurn = true; //Indica a los agentes que es su turno de actuar
        isRunning = false;
        yield return null;
    }

    public IEnumerator CreateSmoke(int turn) //Crea nuevo humo donde es necesario, rastrea las posiciones
    //en las que ya existe el humo para no repetirlas
    {
        string indice = turn.ToString();
        for (int i = 0; i < Board.Smokes[indice].Count; i++)
        {
            List<int> smokePosition = new List<int>(Board.Smokes[indice][i]);
            if(!humo.Any(h => AreListsEqual(h, smokePosition)))
            {
                humo.Add(Board.Smokes[indice][i]);
                Instantiate(bats, new Vector3(-1.5f + (Board.Smokes[indice][i][1] - 1) * 3f, -8f, 17.5f - (Board.Smokes[indice][i][0] - 1) * 3f), Quaternion.Euler(0f, 0f, 0f));
                yield return new WaitForSeconds(0.1f); 
            }
        }
        if(pastHumo.Count != 0) //Código no funcional, busca borrar del tablero elementos que fueron elimminados en el modelo
        {
            Debug.Log("Aqui");
            foreach(List<int> coord in pastHumo)
            {
                Debug.Log("Aqui 2");
                if(!humo.Any(f => AreListsEqual(f, coord)))
                {
                    Debug.Log("Aqui tambien");
                    GameObject[] VampObjects = GameObject.FindGameObjectsWithTag("Bats");
                    foreach (GameObject obj in VampObjects)
                    {
                        if (obj.transform.position == new Vector3(3.5f + (coord[1] - 1) * 3f, 0.1f, 14.5f - (coord[0] - 1) * 3f))
                        {
                            Destroy(obj);
                            Debug.Log("Un objeto instanciado desde 'Bats' fue destruido.");
                        }
                    }
                }
            }
        }
        yield return null;
    }

    public IEnumerator CreateFire(int turn) //Crea nuevo fuego donde es necesario, rastrea las posiciones
    //en las que ya existe el fuego para no repetirlas
    {
        string indice = turn.ToString();
        for (int i = 0; i < Board.Fire[indice].Count; i++)
        {
            List<int> firePosition = new List<int>(Board.Fire[indice][i]);
            if(!fuego.Any(f => AreListsEqual(f, firePosition)))
            {
                fuego.Add(Board.Fire[indice][i]);
                Instantiate(vamp, new Vector3(3.5f + (Board.Fire[indice][i][1] - 1) * 3f, 0.1f, 14.5f - (Board.Fire[indice][i][0] - 1) * 3f), Quaternion.Euler(0f, 0f, 0f));
                yield return new WaitForSeconds(0.1f); 
            }
        }
        if(pastFuego.Count != 0) //Código no funcional, busca borrar del tablero elementos que fueron elimminados en el modelo
        {
            Debug.Log("Aqui, pastFuego tiene " + pastFuego.Count + " elementos.");
            Debug.Log(pastFuego[0]);
            foreach(List<int> coord in pastFuego)
            {
                if(!fuego.Any(f => AreListsEqual(f, coord)))
                {
                    Debug.Log("Aqui tambien");
                    GameObject[] VampObjects = GameObject.FindGameObjectsWithTag("Vamp");
                    foreach (GameObject obj in VampObjects)
                    {
                        if (obj.transform.position == new Vector3(3.5f + (coord[1] - 1) * 3f, 0.1f, 14.5f - (coord[0] - 1) * 3f))
                        {
                            Destroy(obj);
                            Debug.Log("Un objeto instanciado desde 'vamp' fue destruido.");
                        }
                    }
                }
            }
        }
        yield return null;
    }

    public IEnumerator CreateVictims(int turn) //Crea puntos de interés de víctimas
    {
        string indice = turn.ToString();
        for (int i = 0; i < Board.Victimas[indice].Count; i++)
        {
            List<int> POIPosition = new List<int>(Board.Victimas[indice][i]);
            if(!victims.Any(f => AreListsEqual(f, POIPosition)))
            {
                victims.Add(Board.Victimas[indice][i]);
                Instantiate(interest, new Vector3(3.5f + (POIPosition[1] - 1) * 3f, 1f, 14.5f - (POIPosition[0] - 1) * 3f), Quaternion.Euler(90f, 0f, 0f));
                yield return new WaitForSeconds(0.1f); 
            }
        }
        yield return null;
    }

    public IEnumerator CreateFAs(int turn) //Creo puntos de interés de falsas alarmas
    {
        string indice = turn.ToString();
        for (int i = 0; i < Board.FA[indice].Count; i++)
        {
            List<int> POIPosition = new List<int>(Board.FA[indice][i]);
            if(!falsea.Any(f => AreListsEqual(f, POIPosition)))
            {
                falsea.Add(Board.FA[indice][i]);
                Instantiate(interestf, new Vector3(3.5f + (POIPosition[1] - 1) * 3f, 1f, 14.5f - (POIPosition[0] - 1) * 3f), Quaternion.Euler(90f, 0f, 0f));
                yield return new WaitForSeconds(0.1f); 
            }
        }
        yield return null;
    }

    public IEnumerator GetDead(int turn) //Actualiza el número de muertos
    {
        string indice = turn.ToString();
        dead = Board.Muertos[indice];
        yield return null;
    }

    public IEnumerator GetRescued(int turn) //Actualiza el número de rescatados
    {
        string indice = turn.ToString();
        rescued = Board.Rescatados[indice];
        yield return null;
    }

    public IEnumerator GetDP(int turn) //Actualiza el daño al edificio
    {
        string indice = turn.ToString();
        damage = Board.DP[indice];
        GameObject.Find("Damage").GetComponent<DamageUI>().damage = damage;
        yield return null;
    }
}
