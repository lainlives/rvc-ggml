// main_window.cpp — main window with the Song Cover flow as the first tab.
#include "main_window.h"

#include "multi_step_page.h"
#include "song_cover_page.h"
#include "training_page.h"

#include <QTabWidget>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("RVC-GGML UI"));

    auto* tabs = new QTabWidget(this);
    tabs->addTab(new SongCoverPage(tabs), QStringLiteral("Song Cover"));
    tabs->addTab(new MultiStepPage(tabs), QStringLiteral("Multi-step"));
    tabs->addTab(new TrainingPage(tabs), QStringLiteral("Training"));
    setCentralWidget(tabs);
}
