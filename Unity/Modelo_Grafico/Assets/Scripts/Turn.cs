using System.Collections.Generic;
using Newtonsoft.Json;

[System.Serializable]

//Clase para recibir datos del tablero desde el servidor
//Esta información incluye datos sobre paredes; posiciones de las victimas, vampiros, murcielagos y 
//falsas alarmas; junto con el número de muertos, rescatados y puntos de daño al edificio

public class Turn
{
    public Dictionary<string, List<List<string>>> Map = new Dictionary<string, List<List<string>>>();
    public Dictionary<string, List<List<int>>> Victimas = new Dictionary<string, List<List<int>>>();
    public Dictionary<string, List<List<int>>> Fire = new Dictionary<string, List<List<int>>>();
    public Dictionary<string, List<List<int>>> Smokes = new Dictionary<string, List<List<int>>>();
    public Dictionary<string, List<List<int>>> FA = new Dictionary<string, List<List<int>>>();
    public Dictionary<string, int> Muertos = new Dictionary<string, int>();
    public Dictionary<string, int> Rescatados = new Dictionary<string, int>();
    public Dictionary<string, int> DP = new Dictionary<string, int>();
}