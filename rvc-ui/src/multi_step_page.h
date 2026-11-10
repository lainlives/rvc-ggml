// multi_step_page.h — multi-step generation.
//
// Each step runs in isolation: any step's input can be a file ("browse") or any
// other step's output (piped through the shared source registry), and any output
// can be saved anywhere ("Save as...").
#pragma once

#include <QGroupBox>
#include <QObject>
#include <QPair>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QWidget>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QProcess;
class QPushButton;
class QSpinBox;

// Named audio sources shared by all step cards (files + step outputs).
class SourceRegistry : public QObject {
    Q_OBJECT
public:
    void add(const QString& name, const QString& path);
    const QVector<QPair<QString, QString>>& sources() const { return sources_; }

signals:
    void changed();

private:
    QVector<QPair<QString, QString>> sources_;
};

class MultiStepPage;

// One runnable step: separate / split background / convert / pitch-shift / mix.
class StepCard : public QGroupBox {
    Q_OBJECT
public:
    enum class Kind { Separate, SplitBackup, Convert, PitchShift, Mix };

    StepCard(Kind kind, MultiStepPage* owner);

private slots:
    void run();
    void on_input_activated(int index);
    void on_run_finished(int exit_code, int exit_status);
    void save_output(int index);

private:
    void build_ui();
    void refresh_inputs();
    void set_status(const QString& text);
    bool build_invocation(QString* program, QStringList* args, QString* error);
    void handle_success();
    void register_outputs();
    QString input_path(int index) const;
    QString output_path(int index) const;

    Kind kind_;
    MultiStepPage* owner_;
    QVector<QComboBox*> inputs_;
    QVector<QLineEdit*> output_edits_;
    QVector<QString> output_names_;
    QProcess* process_ = nullptr;
    QPushButton* run_button_ = nullptr;
    QLabel* status_label_ = nullptr;
    bool running_ = false;

    // Per-kind widgets.
    QComboBox* model_combo_ = nullptr;
    QComboBox* voice_combo_ = nullptr;
    QComboBox* extractor_combo_ = nullptr;
    QComboBox* f0_combo_ = nullptr;
    QComboBox* device_combo_ = nullptr;
    QDoubleSpinBox* index_rate_spin_ = nullptr;
    QDoubleSpinBox* protect_spin_ = nullptr;
    QDoubleSpinBox* rms_mix_spin_ = nullptr;
    QSpinBox* pitch_spin_ = nullptr;
    QCheckBox* auto_pitch_check_ = nullptr;
    QDoubleSpinBox* threshold_spin_ = nullptr;
    QSpinBox* sid_spin_ = nullptr;
    QVector<QDoubleSpinBox*> gain_spins_;
};

class MultiStepPage : public QWidget {
    Q_OBJECT
public:
    explicit MultiStepPage(QWidget* parent = nullptr);

    SourceRegistry& sources() { return registry_; }
    const QString& base_dir() const { return base_dir_; }

private:
    SourceRegistry registry_;
    QString base_dir_;
};
