using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

// Script que actualiza el texto de un TextMeshPro en cada frame para llevar la cuenta del daño a la
//estructura

public class DamageUI : MonoBehaviour
{
    public TextMeshProUGUI damageText;
    public int damage;
    // Start is called before the first frame update
    void Start()
    {
        damage = 0;
    }

    // Update is called once per frame
    void Update()
    {
        damageText.text = "Damage: " + damage.ToString();
    }
}
