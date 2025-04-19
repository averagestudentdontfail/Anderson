class PricingCommand : public Command {
private:
    PricingService& service_;
    PricingRequest request_;
    PricingResult result_;
        
public:
    PricingCommand(PricingService& service, const PricingRequest& request);
    void execute() override;
    void undo() override;
    const PricingResult& getResult() const;
};