#ifndef COMMAND_H
#define COMMAND_H

#include <string>
#include <memory>
#include <functional>
#include <chrono>
#include <any>
#include <vector>
#include <map>

/**
 * @brief Base command interface
 * 
 * This abstract class defines the interface for all commands
 * in the system, following the Command design pattern.
 */
class Command {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~Command() = default;
    
    /**
     * @brief Execute the command
     * 
     * @return True if the command executed successfully, false otherwise
     */
    virtual bool execute() = 0;
    
    /**
     * @brief Undo the command
     * 
     * @return True if the command was undone successfully, false otherwise
     */
    virtual bool undo() = 0;
    
    /**
     * @brief Check if the command can be undone
     * 
     * @return True if the command can be undone, false otherwise
     */
    virtual bool canUndo() const {
        return false;
    }
    
    /**
     * @brief Get the name of the command
     * 
     * @return Command name
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief Get the command type
     * 
     * @return Command type string
     */
    virtual std::string type() const = 0;
    
    /**
     * @brief Get the parameters of the command
     * 
     * @return Map of parameter names to values
     */
    virtual std::map<std::string, std::any> parameters() const {
        return {};
    }
    
    /**
     * @brief Get the execution time of the command
     * 
     * @return Execution time in microseconds, or 0 if not executed
     */
    int64_t executionTime() const {
        return execution_time_us_;
    }
    
    /**
     * @brief Check if the command execution is deterministic
     * 
     * @return True if the command execution is deterministic, false otherwise
     */
    virtual bool isDeterministic() const {
        return false;
    }
    
    /**
     * @brief Get a description of the command
     * 
     * @return Command description
     */
    virtual std::string description() const {
        return name() + " (" + type() + ")";
    }
    
protected:
    /**
     * @brief Execute the command and measure execution time
     * 
     * @param func The function to execute
     * @return True if the command executed successfully, false otherwise
     */
    bool executeWithTiming(std::function<bool()> func) {
        auto start = std::chrono::high_resolution_clock::now();
        bool result = func();
        auto end = std::chrono::high_resolution_clock::now();
        
        execution_time_us_ = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return result;
    }
    
private:
    int64_t execution_time_us_ = 0;
};

/**
 * @brief Command queue for managing commands
 * 
 * This class manages a queue of commands and their execution.
 */
class CommandQueue {
public:
    /**
     * @brief Constructor
     * 
     * @param max_size Maximum size of the queue (0 for unlimited)
     */
    explicit CommandQueue(size_t max_size = 0) : max_size_(max_size) {}
    
    /**
     * @brief Destructor
     */
    ~CommandQueue() = default;
    
    /**
     * @brief Add a command to the queue
     * 
     * @param command Command to add
     * @return True if the command was added, false otherwise
     */
    bool addCommand(std::shared_ptr<Command> command) {
        if (max_size_ > 0 && commands_.size() >= max_size_) {
            return false;
        }
        
        commands_.push_back(command);
        return true;
    }
    
    /**
     * @brief Execute all queued commands
     * 
     * @return Number of commands executed successfully
     */
    size_t executeAll() {
        size_t success_count = 0;
        
        for (auto& command : commands_) {
            if (command->execute()) {
                success_count++;
            }
        }
        
        // Clear the queue after execution
        commands_.clear();
        
        return success_count;
    }
    
    /**
     * @brief Execute the next command in the queue
     * 
     * @return True if a command was executed successfully, false otherwise
     */
    bool executeNext() {
        if (commands_.empty()) {
            return false;
        }
        
        auto command = commands_.front();
        commands_.erase(commands_.begin());
        
        return command->execute();
    }
    
    /**
     * @brief Get the number of commands in the queue
     * 
     * @return Number of commands
     */
    size_t size() const {
        return commands_.size();
    }
    
    /**
     * @brief Check if the queue is empty
     * 
     * @return True if the queue is empty, false otherwise
     */
    bool empty() const {
        return commands_.empty();
    }
    
    /**
     * @brief Clear the queue
     */
    void clear() {
        commands_.clear();
    }
    
private:
    std::vector<std::shared_ptr<Command>> commands_;
    size_t max_size_;
};

/**
 * @brief Composite command that contains multiple commands
 * 
 * This command executes multiple commands as a single transaction.
 */
class CompositeCommand : public Command {
public:
    /**
     * @brief Constructor
     * 
     * @param name Command name
     */
    explicit CompositeCommand(const std::string& name) : name_(name) {}
    
    /**
     * @brief Add a command to the composite
     * 
     * @param command Command to add
     */
    void addCommand(std::shared_ptr<Command> command) {
        commands_.push_back(command);
    }
    
    /**
     * @brief Execute all commands in the composite
     * 
     * @return True if all commands executed successfully, false otherwise
     */
    bool execute() override {
        return executeWithTiming([this]() {
            for (auto& command : commands_) {
                if (!command->execute()) {
                    // If any command fails, undo the previous ones
                    for (auto it = commands_.rbegin(); it != commands_.rend(); ++it) {
                        if (*it == command) {
                            break;
                        }
                        if ((*it)->canUndo()) {
                            (*it)->undo();
                        }
                    }
                    return false;
                }
            }
            return true;
        });
    }
    
    /**
     * @brief Undo all commands in the composite in reverse order
     * 
     * @return True if all commands were undone successfully, false otherwise
     */
    bool undo() override {
        for (auto it = commands_.rbegin(); it != commands_.rend(); ++it) {
            if ((*it)->canUndo() && !(*it)->undo()) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * @brief Check if the composite can be undone
     * 
     * @return True if at least one command can be undone, false otherwise
     */
    bool canUndo() const override {
        for (const auto& command : commands_) {
            if (command->canUndo()) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief Get the name of the composite command
     * 
     * @return Command name
     */
    std::string name() const override {
        return name_;
    }
    
    /**
     * @brief Get the type of the composite command
     * 
     * @return "Composite"
     */
    std::string type() const override {
        return "Composite";
    }
    
    /**
     * @brief Get the number of commands in the composite
     * 
     * @return Number of commands
     */
    size_t size() const {
        return commands_.size();
    }
    
    /**
     * @brief Check if the composite is empty
     * 
     * @return True if the composite is empty, false otherwise
     */
    bool empty() const {
        return commands_.empty();
    }
    
private:
    std::string name_;
    std::vector<std::shared_ptr<Command>> commands_;
};

#endif // COMMAND_H