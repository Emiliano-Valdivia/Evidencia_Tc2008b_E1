using System.Collections.Generic;
[System.Serializable]

//Clase que crea un diccionario para recibir la información sobre las posiciones de cada agente en cada
//nuevo step

public class AgentTurn 
{
    public Dictionary<string, List<int>> Pos = new Dictionary<string, List<int>>();
}