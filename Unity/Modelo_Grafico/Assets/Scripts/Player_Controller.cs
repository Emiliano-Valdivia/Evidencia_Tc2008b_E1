using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Script para controlar a la cámara indirectamente a través de un jugador invisible

public class Player_Controller : MonoBehaviour
{
    public float speed = 10.0f;
    public float horizontalInput;
    public float forwardInput;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        horizontalInput = Input.GetAxis("Horizontal");
        forwardInput = Input.GetAxis("Vertical");
        transform.Translate(Vector3.forward * Time.deltaTime * speed * forwardInput);
        transform.Translate(Vector3.right * Time.deltaTime * speed * horizontalInput);
    }
}
