using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Destruye un texto 2 segundos después de que aparezca, usado para los popup al tocar un POI

public class PopUp : MonoBehaviour
{
    public float time = 2f;
    // Start is called before the first frame update
    void Start()
    {
        Destroy(gameObject, time);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
