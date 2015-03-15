/////////////////////////////////////////////////////
/// The volume rendering widget on Qt
/// Chun-Ming Chen
/////////////////////////////////////////////////////

#include "cpp_headers.h"
#include "vtk_headers.h"
#include "VRWidget.h"
#include "GL/glut.h"
#include "transfer_func1d.h"

using namespace std;

#define VR_MAX_STEPS 1000
#define FOV 60  // degree

VRWidget :: VRWidget(QWidget *parent)
 : GLVTKWidget(parent)
{
    gpuComm = new CudaGLBase();
    VRParam &vrParam = gpuComm->vrParam;
    vrParam.imgWidth = 256, vrParam.imgHeight=256;
    vrParam.step = 1.f; //.5f;
    vrParam.imgWidth_coalesced = 256;
    vrParam.clippingVisible = 1.f;
    vrParam.dof = 1.f;
    vrParam.d_zBuffer = NULL;
    vrParam.maxSteps = VR_MAX_STEPS;
    vrParam.intensity = 1;
    //vrParam.intensity = 1.f; // embeded in transfer function now

    gpuComm->setUnitLen(this->eyeAngle);

    /// transfer function window
    trfn_window = new Transfer_Func1D(NULL);
    trfn_window->show();
    QObject::connect(trfn_window, SIGNAL(trfn_changed(vector<float>&, vector<float>&)),
                     this, SLOT(on_trfn_changed(vector<float>&, vector<float>&)));

    /// init cuda transfer function    
    g_releaseTrFn();
    g_createTrFn(TrFn_WIDTH, TrFn_WIDTH);

}

VRWidget::~VRWidget()
{
    initialized = false;
    removeData();
    delete gpuComm;

    g_releaseTrFn();

    trfn_window->close();
    delete trfn_window;

}

void VRWidget :: initializeGL()
{
    GLVTKWidget::initializeGL();
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
    case Qt::Key_B:
        draw_boundbox = ! draw_boundbox;
        break;
    }


    updateGL();
}

void VRWidget :: resizeGL(int w, int h)
{
   GLVTKWidget::resizeGL(w, h) ;

   gpuComm->resize(w, h);
}

void VRWidget:: opengl_draw()
{
    if (!initialized)
        return;

    if (this->trfn_dirty){
        this->uploadTrFn2D();
        cudaDeviceSynchronize();
    }

    PRINT("rendering frame started...\n");

    VRParam &vrParam = gpuComm->vrParam;


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
    if (draw_boundbox) {
        glPushMatrix();
        {
            glScalef(vrParam.volBoundry[0],vrParam.volBoundry[1],vrParam.volBoundry[2]);
            glColor3f(0,0,.5);
            glutWireCube(1);
        }
        glPopMatrix();
    }

    gpuComm->checkError("test");
}

void VRWidget::setData(vtkSmartPointer<vtkImageData> data, vtkSmartPointer<vtkImageData> label)
{
    double *range = data->GetScalarRange();
    gpuComm->vrParam.value_min = range[0]-.1;
    gpuComm->vrParam.value_dist = range[1]-range[0];
    printf("Value range: %lf %lf\n", range[0], range[1]);

    int *extent = data->GetExtent();
    int w = extent[1]+1,
        h = extent[3]+1,
        d = extent[5]+1;
    gpuComm->vrParam.maxVolWidthInVoxel = max(max(w,h),d);
    gpuComm->vrParam.volBoundry[0] = w/(float)gpuComm->vrParam.maxVolWidthInVoxel ;
    gpuComm->vrParam.volBoundry[1] = h/(float)gpuComm->vrParam.maxVolWidthInVoxel ;
    gpuComm->vrParam.volBoundry[2] = d/(float)gpuComm->vrParam.maxVolWidthInVoxel ;

    g_createBrickPool(w,h,d);
    g_uploadBrickPool(data->GetScalarPointer(), w, h, d, 0,0,0);

    // label
    g_createLabelPool(w,h,d);
    g_uploadLabelPool(label->GetScalarPointer(), w, h, d, 0,0,0);

    initialized=true;

    trfn_window->emitTranFunc();
}


void VRWidget::uploadTrFn2D()
{
    if (trfn_table.size()==0)
        return ; // not initialized

    g_uploadTrFn(&trfn_table[0], trfn_table.size());
    trfn_dirty = false;
}

void VRWidget::on_trfn_changed(std::vector<float> &color, std::vector<float> &alpha)
{
    int w = color.size()/4;
    int h = alpha.size();
    if (w * h != trfn_table.size())
    {
        g_releaseTrFn();
        g_createTrFn(w, h);
        this->trfn_table_xdim = w;
    }

   // create a 2d table
   trfn_table.clear();
   int x,y;
   for (y=0; y<h; y++)
   {
       for (x=0; x<w; x++)
       {
           trfn_table.push_back(make_float4(color[x*4], color[x*4+1], color[x*4+2], alpha[y]));
       }
   }
   trfn_dirty = true;
   this->updateGL();
}
