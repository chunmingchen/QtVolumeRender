/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

#include "cpp_headers.h"
#include "vtk_headers.h"
#include "VRWidget.h"
#include "GL/glut.h"

using namespace std;

#define VR_MAX_STEPS 500
#define FOV 60  // degree

VRWidget :: VRWidget(QWidget *parent)
 : openGL_VTK_window(parent)
{
    gpuComm = new CudaGLBase();
    VRParam &vrParam = gpuComm->vrParam;
    vrParam.imgWidth = 256, vrParam.imgHeight=256;
    vrParam.step = .5f;
    vrParam.imgWidth_coalesced = 256;
    vrParam.clippingVisible = 1.f;
    vrParam.dof = 1.f;
    vrParam.d_zBuffer = NULL;
    vrParam.maxSteps = VR_MAX_STEPS;
    vrParam.intensity = 1;
    //vrParam.intensity = 1.f; // embeded in transfer function now

    gpuComm->setUnitLen(this->eyeAngle);


}

VRWidget::~VRWidget()
{
    removeData();
    delete gpuComm;

}

void VRWidget :: initializeGL()
{
    openGL_VTK_window::initializeGL();
    int i=0;
    char *s="";
    glutInit(&i, &s);
    glewInit();

    gpuComm->initDevice();

    // shading
    g_setPhongShading();

}

void VRWidget :: keyPressEvent(QKeyEvent* QE)
{
    VRParam &vrParam = gpuComm->vrParam;
    switch (QE->key())
    {
    case  Qt::Key_1:
        //cout<<"Coloring with I11"<<endl;
        break;
    case Qt::Key_Plus:
        vrParam.intensity *= 1.1;
        cout << "Intensity = " << vrParam.intensity << endl;
        break;
    case Qt::Key_Minus:
        vrParam.intensity /= 1.1;
        if (vrParam.intensity < 1e-5) vrParam.intensity = 1e-5;
        cout << "Intensity = " << vrParam.intensity << endl;
        break;
    }


    updateGL();
}

void VRWidget :: resizeGL(int w, int h)
{
   openGL_VTK_window::resizeGL(w, h) ;

   gpuComm->resize(w, h);
}

void VRWidget:: opengl_draw()
{
    if (!initialized)
        return;

    PRINT("rendering frame started...\n");

    VRParam &vrParam = gpuComm->vrParam;


    float s = vrParam.maxVolWidthInVoxel;
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    {
        // shift to volume center
        glTranslatef(-(vrParam.volBoundry[0]*.5), -(vrParam.volBoundry[1]*.5), -(vrParam.volBoundry[2]*.5));

        gpuComm->executeGpuRender();
    }
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // Bounding box
    glPushMatrix();
    {
        glScalef(vrParam.volBoundry[0],vrParam.volBoundry[1],vrParam.volBoundry[2]);
        glColor3f(0,0,.5);
        glutWireCube(1);
    }
    glPopMatrix();

    gpuComm->checkError("test");


}
