// main_window.cpp — main window with the Song Cover flow as the first tab.
#include "main_window.h"

#include "multi_step_page.h"
#include "song_cover_page.h"

#include <QTabWidget>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("RVC-GGML UI"));

    auto* tabs = new QTabWidget(this);
    tabs->addTab(new SongCoverPage(tabs), QStringLiteral("Song Cover"));
    tabs->addTab(new MultiStepPage(tabs), QStringLiteral("Multi-step"));
    // The "Training" tab is planned; the tab widget keeps it easy to add.
    setCentralWidget(tabs);
}
