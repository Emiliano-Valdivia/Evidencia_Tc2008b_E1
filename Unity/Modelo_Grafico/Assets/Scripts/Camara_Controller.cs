using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Clase que controla la cámara para hacer que esta se mueva por el tablero
//La cámara se controla respecto a un jugador invisible y sin colisiones

public class Camara_Controller : MonoBehaviour
{
    public GameObject player;
    private Vector3 offset = new Vector3(0, 17, 0);

    // Update is called once per frame
    void LateUpdate()
    {
        transform.position = player.transform.position + offset;
    }
}