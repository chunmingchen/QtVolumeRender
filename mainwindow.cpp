#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "vtk_headers.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);


}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btnLoad_clicked()
{
    vtkNew<vtkXMLImageDataReader> reader;
    vtkNew<vtkImageCast> caster;
    //reader->SetFileName("/data/flow2/soumya/raw_data/tornado_lambda2_1.vti");
    reader->SetFileName("/data/flow2/isabel_all/Pf05.binLE.raw_corrected_normalized.vti");
    //reader->SetFileName("/data/flow2/soumya/visibility_fields/combined_final1.vti");
    caster->SetInputConnection(reader->GetOutputPort());
    caster->SetOutputScalarTypeToFloat();
    caster->Update();

    this->data = vtkSmartPointer<vtkImageData>::New();
    this->data->DeepCopy(caster->GetOutput()) ;

    this->data->PrintSelf(cout, vtkIndent());
    //this->data->GetPointData()->SetActiveScalars("ImageScalars");

    //reader->SetFileName("/data/flow2/soumya/visibility_fields/combined_final1.vti");
    //reader->Update();
    //this->label = vtkSmartPointer<vtkImageData>::New();
    //this->label->DeepCopy(reader->GetOutput()) ;

    VRWidget *vrwidget = ui->widget;  // look into ui_mainwindow.h
    vrwidget->setData(data);

    vrwidget->updateGL();
}
