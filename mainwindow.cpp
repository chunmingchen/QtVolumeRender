#include "mainwindow.h"
#include "ui_mainwindow.h"


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
    reader->SetFileName("/data/flow2/soumya/raw_data/tornado_lambda2_1.vti");
    reader->Update();
    this->data = reader->GetOutput() ;

    reader->SetFileName("/data/flow2/soumya/visibility_fields/combined_final1.vti");
    reader->Update();
    this->label = reader->GetOutput() ;

    VRWidget *vrwidget = ui->widget;  // look into ui_mainwindow.h
    vrwidget->setData(data);

    vrwidget->updateGL();
}
