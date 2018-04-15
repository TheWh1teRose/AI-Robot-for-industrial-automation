using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System;

public class UnityPythonConector : MonoBehaviour {

    public GameObject traget;
    public GameObject[] random;
    public GameObject[] stay;
    private UdpClient client = null;
    private UdpClient rcvClient = null;
    private Thread receiveThread;
    private float[,,] positionMatrix;
    private int isRestarted = 1;
    private float timer = 0;
    private float sendTimer = 0;
    private bool isRunThread = false;

    public float moveSpeed = 0.35f;

     int horizontal = 0;
     int vertical = 0;
     int height = 0;

    private float x = 60;
    private float y = 60;
    private float h = 60;
    private int smothing = 8;
    //                         x,   y,    h
    private float[] startPositions = {-1.3f,0.5f,1.3f};
    private float[] endPositions   = {1.1f,2.9f,3.7f};
    // Use this for initialization

    Vector3 startPos;

    float finishXMin = -0.8005753f;
    float finishXMax = -0.65f;
    float finishYMin = 2.364033f;
    float finishYMax = 2.534588f;

    int diff = 0;
    int model = 1;
    int ckpt = 3;

    int get = 0;
    int not = 0;
    int didGet = 0; 

    void Start () {
      Application.runInBackground = true;
        startPos = traget.transform.position;

        client = new UdpClient(5002);
        rcvClient = new UdpClient(5003);

        positionMatrix = new float[Convert.ToInt32(x), Convert.ToInt32(y), Convert.ToInt32(h)];
        receiveThread = new Thread(new ThreadStart(RemoteControl));
        receiveThread.Start();
        Debug.Log("test");
    }

	// Update is called once per frame
	void Update () {
        Debug.Log("get:" + get);
        Debug.Log("Not:" + not);
        if (isRunThread == false)
        {
            receiveThread = new Thread(new ThreadStart(RemoteControl));
            receiveThread.Start();
        }
        if (client==null)
        {
            client = new UdpClient(5002);
        }
        if(Input.GetKeyDown("f") && timer > 1){

            traget.transform.position = startPos;
            foreach (GameObject g in random)
            {
                Debug.Log(g.transform.position);
                g.GetComponent<CubeControler>().isSticking = false;
                g.GetComponent<Rigidbody>().useGravity = true;
                g.GetComponent<Random>().random();
            }
            foreach (GameObject g in stay)
            {
                Debug.Log(g.transform.position);
                g.GetComponent<CubeControler>().isSticking = false;
                g.GetComponent<Rigidbody>().useGravity = true;
                g.transform.position = new Vector3(-0.274f, 1.171f, 2.449f);
            }
            isRestarted = 1;
            timer = 0;
            not++;
            didGet = -1;
        }

        if (Input.GetKeyDown("q"))
        {
            diff = 1;
            UDPSendControls();
            client.Close();
            isRunThread = false;
            SceneManager.LoadScene("Scenes/Difficultiy1");
        }
        if (Input.GetKeyDown("w"))
        {
            diff = 2;
            UDPSendControls();
            client.Close();
            isRunThread = false;
            SceneManager.LoadScene("Scenes/Difficultiy3");
        }
        if (Input.GetKeyDown("e"))
        {
            diff = 3;
            UDPSendControls();
            client.Close();
            isRunThread = false;
            SceneManager.LoadScene("Scenes/Difficultiy4_1");
        }


        if (Input.GetKeyDown("r"))
        {
            model = 1;
        }
        if (Input.GetKeyDown("t"))
        {
            model = 2;
        }
        if (Input.GetKeyDown("z"))
        {
            model = 3;
        }

        if (Input.GetKeyDown("p"))
        {
            model = 4;
            ckpt = 3;
        }

        if (Input.GetKeyDown("1"))
        {
            ckpt = 1;
        }
        if (Input.GetKeyDown("2"))
        {
            ckpt = 2;
        }
        if (Input.GetKeyDown("3"))
        {
            ckpt = 3;
        }

        String name = SceneManager.GetActiveScene().name;
        if (name.Equals("Difficultiy1"))
        {
            diff = 1;
        }
        if (name.Equals("Difficultiy3"))
        {
            diff = 2;
        }
        if (name.Equals("Difficultiy4_1"))
        {
            
            diff = 3;
            finishXMin = -0.73f;
            finishXMax = -0.6f;
            finishYMin = 2.45f;
            finishYMax = 2.57f;
        }
        Debug.Log(finishXMin);
        if ((traget.transform.position.x <= finishXMax && traget.transform.position.x >= finishXMin) && (traget.transform.position.z <= finishYMax && traget.transform.position.z >= finishYMin))
        {
            foreach (GameObject cube in random)
            {
                Debug.Log(cube.transform.position);
                cube.GetComponent<CubeControler>().isSticking = false;
                cube.GetComponent<Rigidbody>().useGravity = true;
            }

        }
        if ((traget.transform.position.x <= -1.8f || traget.transform.position.x >= 1.5) || (traget.transform.position.z <= 1.84f || traget.transform.position.z >= 3.2f) || (traget.transform.position.y <= 1.2f || traget.transform.position.y >= 2.8f))
        {
            Debug.Log((traget.transform.position.x <= -1.8f || traget.transform.position.x >= 1.5));
            traget.transform.position = startPos;
            foreach (GameObject g in random)
            {
                Debug.Log(g.transform.position);
                g.GetComponent<CubeControler>().isSticking = false;
                g.GetComponent<Rigidbody>().useGravity = true;
                g.GetComponent<Random>().random();
            }
            foreach (GameObject g in stay)
            {
                Debug.Log(g.transform.position);
                g.GetComponent<CubeControler>().isSticking = false;
                g.GetComponent<Rigidbody>().useGravity = true;
                g.transform.position = new Vector3(-0.274f, 1.171f, 2.449f);
            }
            isRestarted = 1;
            timer = 0;
            not++;
            didGet = -1;
        }

        if (diff==3)
        {
            int onFinish = 0;
            foreach (GameObject cube in random)
            {
                if ((cube.transform.position.x <= finishXMax + 0.1f && cube.transform.position.x >= finishXMin - 0.1f) && (cube.transform.position.z <= finishYMax+0.1f && cube.transform.position.z >= finishYMin-0.1f) && (cube.transform.position.y <= 1.57))
                {
                    onFinish += 1;
                }
            }
            Debug.Log(onFinish);
            if (onFinish == 2)
            {
                traget.transform.position = startPos;
                foreach (GameObject g in random)
                {
                    g.GetComponent<Random>().random();
                }
                isRestarted = 1;
                timer = 0;
                get++;
                didGet = 1;
            }
        }
        if (diff==1)
        {
            int onFinish = 0;
            foreach (GameObject cube in stay)
            {
                if ((cube.transform.position.x <= finishXMax && cube.transform.position.x >= finishXMin) && (cube.transform.position.z <= finishYMax && cube.transform.position.z >= finishYMin) && (cube.transform.position.y <= 1.331f))
                {
                    onFinish += 1;
                }
            }
            Debug.Log(onFinish);
            if (onFinish == 1)
            {
                traget.transform.position = startPos;
                foreach (GameObject g in stay)
                {
                    g.transform.position = new Vector3(-0.274f, 1.171f, 2.449f);
                }
                isRestarted = 1;
                timer = 0;
                get++;
                didGet = 1;
            }
            if ((traget.transform.position.x <= finishXMax && traget.transform.position.x >= finishXMin) && (traget.transform.position.z <= finishYMax && traget.transform.position.z >= finishYMin))
            {
                foreach (GameObject cube in stay)
                {
                    Debug.Log(cube.transform.position);
                    cube.GetComponent<CubeControler>().isSticking = false;
                    cube.GetComponent<Rigidbody>().useGravity = true;
                }
            }
        }
        if (diff==2)
        {
            int onFinish = 0;
            foreach (GameObject cube in random)
            {
                if ((cube.transform.position.x <= finishXMax+0.1f && cube.transform.position.x >= finishXMin-0.1f) && (cube.transform.position.z <= finishYMax+0.1f && cube.transform.position.z >= finishYMin-0.1) && (cube.transform.position.y <= 1.331f))
                {
                    onFinish += 1;
                }
            }
            Debug.Log(onFinish);
            if (onFinish == 1)
            {
                traget.transform.position = startPos;
                foreach (GameObject g in random)
                {
                    g.GetComponent<Random>().random();
                }
                isRestarted = 1;
                timer = 0;
                get++;
                didGet = 1;
            }

        }
        Debug.Log(timer);
        if (timer>=45)
        {
            traget.transform.position = startPos;
            foreach (GameObject g in random)
            {
                Debug.Log(g.transform.position);
                g.GetComponent<CubeControler>().isSticking = false;
                g.GetComponent<Rigidbody>().useGravity = true;
                g.GetComponent<Random>().random();
            }
            foreach (GameObject g in stay)
            {
                Debug.Log(g.transform.position);
                g.GetComponent<CubeControler>().isSticking = false;
                g.GetComponent<Rigidbody>().useGravity = true;
                g.transform.position = new Vector3(-0.274f, 1.171f, 2.449f);
            }
            isRestarted = 1;
            timer = 0;
            not++;
            didGet = -1;
        }

            timer += Time.deltaTime;
        sendTimer += Time.deltaTime;

        Debug.Log("" + diff + ":" + model + ":" + ckpt);

        traget.transform.Translate(moveSpeed * horizontal*Time.deltaTime,
            moveSpeed * vertical * Time.deltaTime,
            moveSpeed * height * Time.deltaTime);

        UDPSendControls();
    }

    private void OnApplicationQuit()
    {
        client.Close();
        isRunThread = false;
    }

    private void UDPSendControls()
    {
        if (sendTimer>0.15)
        {

          int horizontal = 0;
          int vertical = 0;
          int height = 0;
          if(Input.GetAxis("Horizontal")>0){horizontal=1;}
          else if(Input.GetAxis("Horizontal")<0){horizontal=-1;}
          else if(Input.GetAxis("Vertical")>0){vertical=1;}
          else if(Input.GetAxis("Vertical")<0){vertical=-1;}
          else if(Input.GetAxis("Height")>0){height=1;}
          else if(Input.GetAxis("Height")<0){height=-1;}

            //Debug.Log(isRestarted);
                
            string controls = horizontal + ":"
            + -vertical + ":"
            + -height + ":"  + getGrab();

            //Debug.Log(controls);

            IPEndPoint endpoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5002);
            byte[] sendBytes = Encoding.ASCII.GetBytes(controls + "$" + isRestarted + "$" + getPositionInMatrix() + "$" +
                diff + "$" + model + "$" + ckpt + "$" + didGet);
            didGet = 0;
            isRestarted = 0;
            client.Send(sendBytes, sendBytes.Length, endpoint);
            sendTimer=0;
        }
    }

    private void RemoteControl()
    {
        isRunThread = true;
        while (isRunThread)
          {
            IPEndPoint endpoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5003);
            Byte[] receiveBytes = rcvClient.Receive(ref endpoint);
            string returnData = Encoding.ASCII.GetString(receiveBytes);
            if(returnData[0].Equals('1')){horizontal=1;}
            if(returnData[2].Equals('1')){horizontal=-1;}
            if(returnData[4].Equals('1')){vertical=1;}
            if(returnData[6].Equals('1')){vertical=-1;}
            if(returnData[8].Equals('1')){height=1;}
            if(returnData[10].Equals('1')){height=-1;}

            if(returnData[0].Equals('0')&&returnData[2].Equals('0')){horizontal=0;}
            if(returnData[4].Equals('0')&&returnData[6].Equals('0')){vertical=0;}
            if(returnData[8].Equals('0')&&returnData[10].Equals('0')){height=0;}
            //if(returnData[6].Equals('1')){horizontal=1;}
            //Sif(returnData[7].Equals('1')){horizontal=1;}
            //Debug.Log(returnData);
            Debug.Log("TEst");
          }
        return;
    }

    private String getPositionInMatrix()
    {
        //get the postions from the target. NOTE: some dimentions are switcht for realism
        float targetX = traget.transform.position.x;
        float targetY = traget.transform.position.z;
        float targetH = traget.transform.position.y;

        float distanceToStartX = targetX - startPositions[0];
        float distanceToStartY = targetY - startPositions[1];
        float distanceToStartH = targetH - startPositions[2];

        int positionInPosiotinMatrixX = Convert.ToInt32(Math.Round(distanceToStartX / ((endPositions[0] - startPositions[0]) / x)));
        int positionInPosiotinMatrixY = Convert.ToInt32(Math.Round(distanceToStartY / ((endPositions[1] - startPositions[1]) / y)));
        int positionInPosiotinMatrixH = Convert.ToInt32(Math.Round(distanceToStartH / ((endPositions[2] - startPositions[2]) / h)));

        //Debug.Log(positionInPosiotinMatrixX + ":" + positionInPosiotinMatrixY + ":" + positionInPosiotinMatrixH);

        return (positionInPosiotinMatrixX + ":" + positionInPosiotinMatrixY + ":" + positionInPosiotinMatrixH);

    }

    private int getGrab()
    {
        if (Input.GetKey("g"))
        {
            return 1;
        }else if (Input.GetKey("b"))
        {
            return -1;
        }
        return 0;
    }
}
