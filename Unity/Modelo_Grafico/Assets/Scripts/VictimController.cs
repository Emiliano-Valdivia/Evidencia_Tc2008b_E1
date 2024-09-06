using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VictimController : MonoBehaviour
{
    public int dead;
    public int rescued;
    public int damage;
    public GameObject temp;
    public GameOverScreen GameOverScreen;
    public GameOverScreen VictoryScreen;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        //Actualiza  variables basándose en la información actual del tablero
        dead = GameObject.Find("BoardManager").GetComponent<Tablero>().dead;
        rescued = GameObject.Find("BoardManager").GetComponent<Tablero>().rescued;
        damage = GameObject.Find("BoardManager").GetComponent<Tablero>().damage;
        //Verifica número de muertos, hace visible una cruz para cada uno; causa un Game Over al cuarto
        if(dead == 1)
        {
            temp = GameObject.Find("cross-wood");
            temp.transform.position = new Vector3(temp.transform.position.x, 0.1f, temp.transform.position.z);
        }
        if(dead == 2)
        {
            temp = GameObject.Find("cross-wood (1)");
            temp.transform.position = new Vector3(temp.transform.position.x, 0.1f, temp.transform.position.z);
        }
        if(dead == 3)
        {
            temp = GameObject.Find("cross-wood (3)");
            temp.transform.position = new Vector3(temp.transform.position.x, 0.1f, temp.transform.position.z);
        }
        if(dead >= 4)
        {
            temp = GameObject.Find("cross-wood (2)");
            temp.transform.position = new Vector3(temp.transform.position.x, 0.1f, temp.transform.position.z);
            GameOver();
        }
        //Verifica número de daño; causa un Game Over si es 24
        if(damage >= 24)
        {
            GameOver();
        }
        //Verifica número de rescatados, hace visible un personaje para cada uno; causa una victoria al séptimo
        if(rescued == 1)
        {
            temp = GameObject.Find("character-female-c");
            temp.transform.position = new Vector3(temp.transform.position.x, 0.1f, temp.transform.position.z);
        }
        if(rescued == 2)
        {
            temp = GameObject.Find("character-female-b");
            temp.transform.position = new Vector3(temp.transform.position.x, 0.1f, temp.transform.position.z);
        }
        if(rescued == 3)
        {
            temp = GameObject.Find("character-female-e");
            temp.transform.position = new Vector3(temp.transform.position.x, 0.1f, temp.transform.position.z);
        }
        if(rescued == 4)
        {
            temp = GameObject.Find("character-female-d");
            temp.transform.position = new Vector3(temp.transform.position.x, 0.1f, temp.transform.position.z);
        }
        if(rescued == 5)
        {
            temp = GameObject.Find("character-male-d");
            temp.transform.position = new Vector3(temp.transform.position.x, 0.1f, temp.transform.position.z);
        }
        if(rescued == 6)
        {
            temp = GameObject.Find("character-male-b");
            temp.transform.position = new Vector3(temp.transform.position.x, 0.1f, temp.transform.position.z);
        }
        if(rescued >= 7)
        {
            temp = GameObject.Find("character-male-f");
            temp.transform.position = new Vector3(temp.transform.position.x, 0.1f, temp.transform.position.z);
            Victory();
        }
    }

    public void GameOver() //Muestra pantalla de Game Over
    {
        GameOverScreen.Setup();
    }

    public void Victory() //Muestra pantalla de Victoria
    {
        VictoryScreen.Setup();
    }
}
