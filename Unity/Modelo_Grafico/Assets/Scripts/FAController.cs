using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FAController : MonoBehaviour
{
    public GameObject popup;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    //Cuando se detecta una colisión, se destruye el objeto y se muestra un texto indicando que se 
    //encontró una falsa alarma

    void OnTriggerEnter(Collider other)
    {
        Instantiate(popup, new Vector3(transform.position.x, 1f, transform.position.z), Quaternion.Euler(90f, 0f, 0f));
        Destroy(gameObject);
    }
}
